"""Run training, then test the validation-selected checkpoint in a fresh process."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run_child(command, *, cwd, stdout=None, stderr=None):
    """Wait for one stage and forward stop requests to its entire worker group."""
    with subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr,
                          start_new_session=os.name == 'posix') as child:
        interrupted = []
        def forward(signum, frame):
            interrupted.append(signum)
            if child.poll() is None:
                if os.name == 'posix':
                    os.killpg(child.pid, signum)
                else:
                    child.send_signal(signum)
        previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            code = child.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        if interrupted:
            raise RuntimeError('Pipeline was interrupted; no following stage will start')
        if code:
            raise subprocess.CalledProcessError(code, command)


def require_completed_training(run_directory, epochs):
    summary = read_json(run_directory / 'run_summary.json')
    run = read_json(run_directory / 'run.json')
    if (summary.get('status') != 'complete' or summary.get('partial_dataset') is not False
            or summary.get('stop_reason') != 'epochs_completed'
            or summary.get('state', {}).get('epoch') != epochs
            or run.get('config', {}).get('epochs') != epochs
            or not summary.get('run_signature')
            or summary['run_signature'] != run.get('run_signature')
            or not (run_directory / 'best.pt').is_file()):
        raise RuntimeError('Training did not complete the full configured run; automatic test was not started')


def experiment_paths(root, scene=None):
    root = Path(root)
    if scene is not None:
        if scene not in ('medical', 'finance', 'hr'):
            raise ValueError('Unknown scene')
        train_config=root / f'configs/experiment_v6_{scene}_no_negatives.json'
        test_config=root / f'configs/evaluation_v2_{scene}.json'
        run=root / (read_json(train_config)['output_directory'] if train_config.exists()
                    else f'runs/soft_{scene}_v2_clean_seed17_100epochs')
        test=root / (read_json(test_config)['output_directory'] if test_config.exists()
                     else f'runs/test_{scene}_v2_clean_best')
        return train_config,test_config,run,test,root/'logs'/(test.name+'.log')
    return (root / 'configs/experiment_v5_multiscene_no_negatives.json',
            root / 'configs/evaluation_multiscene_v1.json',
            root / 'runs/soft_multiscene_v1_seed17_100epochs',
            root / 'runs/test_multiscene_v1_best', root / 'logs/test_multiscene_v1_best.log')


def run_experiment(root, gpus, *, resume=False, run_directory=None, epoch_validation=None, scene=None):
    root = Path(root)
    train_config, test_config, default_run, test_directory, test_log = experiment_paths(root, scene)
    if not test_config.is_file():
        raise FileNotFoundError(f'Missing evaluation config: {test_config}')
    run_directory = Path(run_directory) if run_directory is not None else default_run
    if scene is not None:
        train_cfg, test_cfg = read_json(train_config), read_json(test_config)
        if any(c.get('scene') != scene or c.get('record_form') is not None for c in (train_cfg, test_cfg)):
            raise ValueError('Scene training and evaluation configs must select the same scene and all its forms')
        if train_cfg['dataset_directory'] != test_cfg['dataset_directory']:
            raise ValueError('Scene train/test dataset mismatch')
        if resume and read_json(run_directory / 'run.json')['config'] != train_cfg:
            # CLI expands defaults; compare with the same loader used by training.
            sys.path.insert(0, str(root))
            from medical_kv.config import load_config
            if read_json(run_directory / 'run.json')['config'] != load_config(train_cfg, {'output_directory': str(run_directory)}):
                raise ValueError('Cannot resume a different dataset/config as a scene run')
    if test_directory.exists() or test_log.exists():
        raise FileExistsError('Test output/log already exists; refusing to overwrite a previous test')
    epochs = read_json(train_config)['epochs']
    train_command = [sys.executable, '-u', '-m', 'medical_kv', 'train-soft',
                     '--config', str(train_config), '--gpus', gpus,
                     '--output-directory', str(run_directory)]
    if resume:
        train_command += ['--resume', str(run_directory / 'latest.pt')]
    if epoch_validation is not None:
        train_command.append('--epoch-validation' if epoch_validation else '--no-epoch-validation')
    print(f'PIPELINE | training | scene={scene or "joint"} | GPUs={gpus} | epochs={epochs}', flush=True)
    run_child(train_command, cwd=root)
    summary = read_json(run_directory / 'run_summary.json')
    if summary.get('stop_reason') in ('paused', 'interrupted'):
        print('PIPELINE | paused | latest.pt saved; automatic test deferred until recovery completes', flush=True)
        return
    require_completed_training(run_directory, epochs)
    # The training subprocess has exited, releasing its model and CUDA allocations.
    print(f'PIPELINE | training complete | testing best.pt | GPUs={gpus}', flush=True)
    print(f'PIPELINE | test log: {test_log}', flush=True)
    test_log.parent.mkdir(parents=True, exist_ok=True)
    test_command = [sys.executable, '-u', '-m', 'medical_kv', 'evaluate',
                    '--config', str(test_config), '--gpus', gpus,
                    '--condition', 'soft', '--split', 'test',
                    '--checkpoint', str(run_directory / 'best.pt'),
                    '--output-directory', str(test_directory)]
    with test_log.open('x', encoding='utf-8') as stream:
        run_child(test_command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
    print(f'PIPELINE | complete | predictions: {test_directory / "predictions.csv"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--resume', action='store_true', help='Resume latest.pt, then automatically test on completion')
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--scene', choices=['medical', 'finance', 'hr'])
    parser.add_argument('--epoch-validation', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    try:
        run_experiment(Path(__file__).resolve().parents[1], args.gpus, resume=args.resume,
                       run_directory=args.run_root, epoch_validation=args.epoch_validation, scene=args.scene)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'PIPELINE | failed or incomplete | {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
