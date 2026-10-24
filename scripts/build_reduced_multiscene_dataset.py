"""Remove selected supervision targets while retaining exact input text and split groups."""
import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

from multiscene_common import ROOT, file_hash, write_json


EXCLUDED = frozenset(('phone', 'sender_account', 'recipient_account'))


def reduce_record(original, dataset_id):
    record = copy.deepcopy(original)
    record['dataset_id'] = dataset_id
    for key in ('fields', 'field_annotations'):
        record[key] = {k: v for k, v in record[key].items() if k not in EXCLUDED}
    for key in ('qa', 'training_qa', 'evaluation_qa'):
        record[key] = [q for q in record[key] if q['field'] not in EXCLUDED and q.get('group') not in EXCLUDED]
    record['native_target_count'] = sum(a['origin'] == 'native' for a in record['field_annotations'].values())
    # Keep input, augmentation/source evidence, IDs and all original isolation groups.
    return record


def build(config_path):
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if set(cfg['excluded_target_fields']) != EXCLUDED:
        raise ValueError('This version removes exactly phone, sender_account and recipient_account')
    source, target = ROOT / cfg['source_directory'], ROOT / cfg['output_directory']
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite dataset: {target}')
    # Verify the frozen parent before deriving the new version.
    hashes = json.loads((source / 'artifact_hashes.json').read_text(encoding='utf-8'))
    for name, expected in hashes.items():
        if file_hash(source / name) != expected:
            raise ValueError(f'Parent artifact changed: {name}')
    shutil.copytree(source, target)
    for path in sorted(target.rglob('*.jsonl')):
        if path.name == 'selected_native_sources.jsonl':
            continue
        temporary = path.with_suffix('.jsonl.tmp')
        with path.open(encoding='utf-8') as stream, temporary.open('w', encoding='utf-8', newline='\n') as out:
            for line in stream:
                record = reduce_record(json.loads(line), cfg['dataset_id'])
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
        temporary.replace(path)
    build_cfg = json.loads((target / 'build_config.json').read_text(encoding='utf-8'))
    build_cfg.update(dataset_id=cfg['dataset_id'], output_directory=cfg['output_directory'],
                     target_reduction=cfg)
    write_json(target / 'build_config.json', build_cfg)
    manifest = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
    manifest.update(dataset_id=cfg['dataset_id'], prepared_date='2026-10-23',
                    target_policy={'excluded_fields': sorted(EXCLUDED), 'private_prefix_unchanged': True},
                    parent_dataset={'dataset_id': json.loads((source / 'manifest.json').read_text(encoding='utf-8'))['dataset_id'],
                                    'manifest_sha256': file_hash(source / 'manifest.json'),
                                    'split_sha256': {s: hashes[s + '.jsonl'] for s in ('train', 'validation', 'test')}})
    manifest['target_fields'] = {s: [f for f in fs if f not in EXCLUDED] for s, fs in manifest['target_fields'].items()}
    manifest['reproduction_code_sha256']['scripts/build_reduced_multiscene_dataset.py'] = file_hash(Path(__file__))
    write_json(target / 'manifest.json', manifest)
    templates = json.loads((target / 'question_templates.json').read_text(encoding='utf-8'))
    # Template artifact is metadata; recursively remove excluded field entries.
    def filter_templates(value):
        if isinstance(value, dict):
            return {k: filter_templates(v) for k, v in value.items() if k not in EXCLUDED}
        if isinstance(value, list):
            return [filter_templates(v) for v in value if not isinstance(v, dict)
                    or (v.get('field') not in EXCLUDED and v.get('group') not in EXCLUDED)]
        return value
    write_json(target / 'question_templates.json', filter_templates(templates))
    from validate_multiscene_dataset import validate
    report = validate(target, write=True)
    if report['status'] != 'PASS':
        raise ValueError(report)
    subprocess.run([sys.executable, str(ROOT / 'scripts/audit_multiscene_tokens.py'),
                    '--directory', cfg['output_directory']], cwd=ROOT, check=True)
    print(json.dumps({'status': 'complete', 'directory': str(target), 'excluded_fields': sorted(EXCLUDED)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/dataset_multiscene_v2_reduced.json')
    args = parser.parse_args()
    build(ROOT / args.config)
