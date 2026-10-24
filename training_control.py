"""Pause and recover the shared-prefix multiscene run without loading a model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from filelock import FileLock
from medical_kv.control import request_pause, training_status, wait_paused, write_json
from medical_kv.launch import parse_gpus

ROOT = Path(__file__).resolve().parent


def recover(root, gpus, *, epoch_validation=None):
    root = Path(root).resolve()
    ids = parse_gpus(','.join(gpus).replace(' ', ''))
    # Serialize concurrent recovery commands and refuse to steal a live run.
    with FileLock(str(root / '.recovery.lock'), timeout=0):
        status = training_status(root)
        if status['running']:
            raise RuntimeError('Training still holds the run lock; wait for pause to finish')
        if not status['checkpoint_exists']:
            raise FileNotFoundError(f'Missing checkpoint: {root / "latest.pt"}')
        if status['status'] == 'complete':
            raise RuntimeError('Training is already complete; use the evaluation command')
        info = json.loads((root / 'run.json').read_text(encoding='utf-8'))
        if info.get('method') != 'soft' or info['config'].get('question_protocol') != 'multiscene_privacy_v1':
            raise ValueError('This recovery command supports the multiscene soft-training pipeline')
        if os.name != 'posix':
            raise RuntimeError('Background recovery runs on the Linux CUDA server')
        scene = info['config'].get('scene')
        # The launcher chooses matching fixed configs/output names for each scene.
        sys.path.insert(0, str(ROOT / 'scripts'))
        from run_multiscene_experiment import experiment_paths
        _, _, _, test_directory, test_log = experiment_paths(ROOT, scene)
        if test_directory.exists() or test_log.exists():
            raise FileExistsError('Test output already exists; refusing to overwrite it')
        # Refuse to start a second pipeline while an earlier recovery is loading.
        from medical_kv.control import process_start
        process_file = root / 'recovery_process.json'
        if process_file.is_file():
            previous = json.loads(process_file.read_text(encoding='utf-8'))
            if previous.get('start_ticks') and process_start(previous['pid']) == previous['start_ticks']:
                raise RuntimeError('A recovery pipeline is already active')
        log = ROOT / 'logs' / ('recover_multiscene_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', str(ROOT / 'scripts/run_multiscene_experiment.py'),
                   '--gpus', ','.join(map(str, ids)), '--resume', '--run-root', str(root)]
        if scene is not None:
            command += ['--scene', scene]
        if epoch_validation is not None:
            command.append('--epoch-validation' if epoch_validation else '--no-epoch-validation')
        with log.open('x', encoding='utf-8') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        payload = {'pid': child.pid, 'start_ticks': process_start(child.pid),
                   'gpus': ids, 'log': str(log), 'run_root': str(root), 'epoch_validation_override': epoch_validation}
        write_json(process_file, payload)
        return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('pause', 'status', 'resume', 'recover'):
        command = commands.add_parser(name)
        command.add_argument('--run-root', type=Path, required=True)
        if name == 'pause':
            command.add_argument('--wait', action='store_true')
            command.add_argument('--timeout', type=float, default=600)
        elif name in ('resume', 'recover'):
            command.add_argument('--epoch-validation', action=argparse.BooleanOptionalAction, default=None,
                                 help='Override per-epoch validation; final validation is always retained')
            command.add_argument('--gpus', nargs='+', required=True,
                                 help='Physical GPU IDs, e.g. 0 1 2 3 or 0,1,2,3')
    args = parser.parse_args(argv)
    try:
        if args.command == 'pause':
            print(json.dumps(request_pause(args.run_root, ROOT), ensure_ascii=False), flush=True)
            result = wait_paused(args.run_root, args.timeout) if args.wait else {'status': 'pause_requested'}
        elif args.command == 'status':
            result = training_status(args.run_root)
        else:
            result = recover(args.run_root, args.gpus,
                             **({'epoch_validation': args.epoch_validation} if args.epoch_validation is not None else {}))
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Training control: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
