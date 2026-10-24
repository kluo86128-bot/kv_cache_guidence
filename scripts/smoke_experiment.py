"""Exercise public CLI paths with actual dev records and random CPU-only Qwen3."""
import argparse
import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from medical_kv.backend import Backend
    from medical_kv.cli import main as cli, preflight, tiny_backend
    from medical_kv.config import load_config
    from medical_kv.data import dataset_identity, load_split
    from medical_kv.io import code_identity
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-directory', default='runs/implementation_checks/cli_smoke')
    args = parser.parse_args()
    out = Path(args.output_directory)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    with (out / 'cli.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        cli(['inspect'])
        cli(['preflight', '--tiny'])
        soft = out / 'soft'
        cli(['train-soft', '--tiny', '--development', '--train-limit', '2', '--validation-limit', '1',
             '--epochs', '2', '--max-new-tokens', '2', '--output-directory', str(soft)])
        assert (soft / 'loss_curve.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        import csv
        validations = [json.loads(line) for line in (soft / 'validation.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(validations) == 2
        for validation in validations:
            with (soft / validation['predictions_csv']).open(encoding='utf-8-sig', newline='') as stream:
                answers = list(csv.DictReader(stream))
            assert len(answers) == 7 and {int(row['epoch']) for row in answers} == {validation['epoch']}
            assert validation['metrics']['qa_pairs'] == 6
        assert (soft / 'final.pt').exists()
        for condition in ('soft', 'random', 'blocked', 'full', 'manual', 'none'):
            command = ['evaluate', '--tiny', '--development', '--split', 'dev', '--limit', '2',
                       '--max-new-tokens', '2', '--condition', condition,
                       '--output-directory', str(out / ('eval_' + condition))]
            if condition in ('soft', 'random', 'blocked', 'full'):
                command += ['--checkpoint', str(soft / 'best.pt')]
            cli(command)
        cli(['compare', '--a', str(out / 'eval_soft/predictions.jsonl'),
             '--b', str(out / 'eval_random/predictions.jsonl'), '--output-directory', str(out / 'comparison')])
        cfg = load_config('configs/experiment_v1.json', {'device': 'cpu', 'dtype': 'float32', 'max_new_tokens': 2})
        backend = tiny_backend(cfg)
        local_model = out / 'local_tiny_model'
        backend.model.save_pretrained(local_model)
        backend.tokenizer.save_pretrained(local_model)
        cfg['model_name_or_path'] = str(local_model)
        restored = Backend.load(cfg, dataset_identity(cfg)['tokenizer_sha256'])
        loaded = preflight(restored, cfg, load_split(cfg, 'dev', 1))
        assert loaded['status'] == 'PASS'
    conditions = {}
    for condition in ('soft', 'random', 'blocked', 'full', 'manual', 'none'):
        summary = json.loads((out / f'eval_{condition}/summary.json').read_text(encoding='utf-8'))
        assert summary['status'] == 'complete' and summary['records'] == 2
        assert summary['metadata']['tiny_implementation_check'] is True
        assert summary['metadata']['generation']['max_new_tokens'] == 2
        assert summary['metadata']['evaluation_code_sha256'] == code_identity()
        with (out / f'eval_{condition}/predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
            assert len(list(csv.DictReader(stream))) == 14
        assert summary['qa_pairs'] == 12 and summary['birth_date_mode'] == 'parts'
        conditions[condition] = {'status': 'PASS', 'records': 2, 'answers': 12, 'csv_rows': 14}
    report = {'status': 'PASS', 'implementation_only': True, 'model': 'random tiny Qwen3; CPU',
              'source_records': 'dev only; training subset dev[:2], selection dev[64:65], eval dev[:2]',
              'soft_training': 'PASS', 'hard_training': 'disabled', 'preflight': 'PASS',
              'local_AutoModel_AutoTokenizer_loading': 'PASS', 'paired_comparison': 'PASS',
              'loss_curve': 'PASS',
              'per_epoch_validation_csv': 'PASS', 'standalone_evaluation_csv': 'PASS',
              'conditions': conditions, 'code_sha256': code_identity(), 'formal_4b_run': False}
    (out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'verification_report': str(out / 'verification.json')}, indent=2))


if __name__ == '__main__':
    main()
