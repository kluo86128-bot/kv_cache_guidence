"""Build the approved 14-field HR view with isolated, reusable event donors.

No model is loaded. Frozen v2 data, other scenes and old experiments are kept.
Downloaded files are revision-pinned, hash-checked inert CSV/JSON snapshots.
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import subprocess
import sys
import urllib.request
import uuid
from pathlib import Path

from hr_augmentation import read_donors, group_and_partition, augment_rows, connect_augmented_groups
from multiscene_common import ROOT, SCENES, file_hash, write_json


def fetch_sources(cfg):
    cache = ROOT / cfg['source_cache']
    cache.mkdir(parents=True, exist_ok=True)
    for source in cfg['sources']:
        path = cache / source['filename']
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.download')
            with urllib.request.urlopen(source['url'], timeout=60) as response:
                temporary.write_bytes(response.read())
            if file_hash(temporary) != source['sha256']:
                raise ValueError('Downloaded source hash mismatch: ' + source['name'])
            temporary.replace(path)
        if file_hash(path) != source['sha256']:
            raise ValueError('Cached source changed: ' + source['name'])
    write_json(cache / 'sources.json', cfg['sources'])
    return cache


def build(config_path, *, rebuild=False):
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    source, target = ROOT / cfg['source_directory'], ROOT / cfg['output_directory']
    if target.exists():
        if not rebuild:
            raise FileExistsError('Existing dataset will not be overwritten without --rebuild: ' + str(target))
        previous = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
        if previous['dataset_id'] != cfg['dataset_id'] or target.resolve() == source.resolve():
            raise ValueError('Rebuild target is not this derived dataset')
        for name, sha in json.loads((target / 'artifact_hashes.json').read_text(encoding='utf-8')).items():
            if file_hash(target / name) != sha:
                raise ValueError('Derived dataset edited after publication; preserve it before rebuilding: ' + name)
    parent_hashes = json.loads((source / 'artifact_hashes.json').read_text(encoding='utf-8'))
    for name, sha in parent_hashes.items():
        if file_hash(source / name) != sha:
            raise ValueError('Frozen parent changed: ' + name)
    cache = fetch_sources(cfg)
    donors, rejected = read_donors(cache)
    donors = group_and_partition(donors, cfg['seed'], cfg['near_duplicate_jaccard'])
    rows = {}
    for split in ('train', 'validation', 'test'):
        rows[split] = [json.loads(line) for line in (source / (split + '.jsonl')).open(encoding='utf-8')]
    hr = [r for records in rows.values() for r in records if r['scene'] == 'hr']
    augmented = augment_rows(hr, donors, cfg['hr_fields'], cfg['coverage_per_field'], cfg['seed'])
    connect_augmented_groups(augmented)
    replacements = {r['record_id']: r for r in augmented}
    for r in augmented:
        r['dataset_id'] = cfg['dataset_id']
    stage = target.parent / ('_' + target.name + '_staging_' + uuid.uuid4().hex[:8])
    # Parent convenience archives are not training artifacts of this release.
    shutil.copytree(source, stage, ignore=shutil.ignore_patterns('*.zip'))
    # Each parent row keeps its ID/split/order. Non-HR bytes remain identical.
    for split in rows:
        streams = {}
        try:
            with (stage / (split + '.jsonl')).open('w', encoding='utf-8', newline='\n') as out:
                for line in (source / (split + '.jsonl')).open(encoding='utf-8'):
                    r = json.loads(line)
                    if r['scene'] == 'hr':
                        r = replacements[r['record_id']]
                        line = json.dumps(r, ensure_ascii=False) + '\n'
                        form = r['record_form']
                        if form not in streams:
                            streams[form] = (stage / 'hr' / form / (split + '.jsonl')).open('w', encoding='utf-8', newline='\n')
                        streams[form].write(line)
                    out.write(line)
        finally:
            for stream in streams.values():
                stream.close()
    # Replace only HR preview records. Never generate a preview from held-out data.
    previews = [json.loads(line) for line in (source / 'preview_30.jsonl').open(encoding='utf-8')]
    with (stage / 'preview_30.jsonl').open('w', encoding='utf-8', newline='\n') as out:
        for r in previews:
            out.write(json.dumps(replacements.get(r['record_id'], r), ensure_ascii=False) + '\n')
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    manifest.update(dataset_id=cfg['dataset_id'], prepared_date=cfg['prepared_date'],
                    hr_augmentation_revision='hr_balanced_events_v1',
                    parent_dataset={'directory': cfg['source_directory'], 'manifest_sha256': file_hash(source / 'manifest.json'),
                                    'split_sha256': {s: parent_hashes[s + '.jsonl'] for s in rows}})
    manifest['target_fields']['hr'] = cfg['hr_fields']
    excluded_hr = sorted(set(SCENES['hr']) - set(cfg['hr_fields']))
    manifest['target_policy'].update(hr_retained_fields=cfg['hr_fields'],
        excluded_fields=sorted(set(manifest['target_policy']['excluded_fields']) | set(excluded_hr)),
        hr_original_native_input_preserved=True,
        hr_removed_targets_may_remain_in_original_native_text=True,
        hr_added_blocks_rerendered_without_provenance_markers=True)
    manifest['augmentation_sources'] = cfg['sources']
    manifest['hr_coverage_per_field'] = cfg['coverage_per_field']
    manifest['limitations'] = [s for s in manifest.get('limitations', []) if 'Old medical_kv four-field loader' not in s]
    manifest['limitations'] += [
        'HR events are constructed associations; repeated donors within a split do not constitute independent real-world events.',
        'Existing native partial events are preserved; their synthetic completion is not an assertion of original-source facts.',
        'Source scenario roles used in new complaints are constructed, not an adjudication or verified accusation.',
        'Automatic structural/isolation checks are not a complete manual semantic review.',
        'Common short answers may occur in multiple splits; exact answer overlap is reported separately.',
        'Reused HR donors may create large connected groups; development views may have insufficient independent groups.']
    manifest['reproduction_code_sha256'].update({str(p): file_hash(ROOT / p) for p in (
        'scripts/build_hr_balanced_dataset.py', 'scripts/hr_augmentation.py', 'scripts/validate_hr_balanced_dataset.py')})
    write_json(stage / 'manifest.json', manifest)
    build_cfg = json.loads((source / 'build_config.json').read_text(encoding='utf-8'))
    build_cfg.update(dataset_id=cfg['dataset_id'], output_directory=cfg['output_directory'], hr_augmentation=cfg)
    write_json(stage / 'build_config.json', build_cfg)
    write_json(stage / 'hr_augmentation_config.json', cfg)
    with (stage / 'hr_source_events.jsonl').open('w', encoding='utf-8', newline='\n') as out:
        for donor in donors:
            out.write(json.dumps(donor, ensure_ascii=False) + '\n')
    write_json(stage / 'hr_source_inventory.json', {
        'sources': cfg['sources'], 'extraction_rejections': rejected,
        'donors_by_split': {s: dict(collections.Counter(d['source'] + '/' + d['kind'] for d in donors if d['assigned_split'] == s)) for s in rows},
        'families_by_split': {s: {kind: len({d['event_group'] for d in donors if d['assigned_split'] == s and d['kind'] == kind})
                                  for kind in ('leave', 'complaint')} for s in rows},
        'realigns_policy': 'Snapshot audited but not used as a donor; nonstandard reuse terms require clarification.',
        'eeoc_policy': 'Not scraped or used in this release; reserve as a future scenario source.'})
    from validate_multiscene_dataset import validate
    report = validate(stage, write=True)
    if report['status'] != 'PASS':
        raise ValueError(report)
    subprocess.run([sys.executable, str(ROOT / 'scripts/audit_multiscene_tokens.py'),
                    '--directory', str(stage)], check=True, cwd=ROOT)
    stats = json.loads((stage / 'statistics.json').read_text(encoding='utf-8'))
    stats['record_group_ids'] = stats.pop('independent_connected_groups')
    stats['group_count_scope'] = 'Non-HR parent records/groups are frozen; HR source components were recomputed. This is not a newly computed global independent-event count.'
    write_json(stage / 'statistics.json', stats)
    dictionary = json.loads((stage / 'data_dictionary.json').read_text(encoding='utf-8'))
    dictionary['native_text'] = 'First native_text_length characters retain the cleaned v2 input. Original publisher text and deletion audit remain preserved.'
    dictionary['augmentation_blocks'] = 'Business facts only in model text. Source IDs, licenses, split ownership and synthetic derivations are metadata outside private_prefix and questions.'
    dictionary['hr_augmentation'] = {'fields': cfg['hr_fields'], 'coverage': cfg['coverage_per_field'],
        'group_id': 'Recomputed HR connected components join parent groups and all reused event donors; parent_group_id retains the previous group.',
        'input_cache': 'Rebuild into a new cache directory using --dataset-directory and --scene hr.',
        'rebuild': 'python scripts/build_hr_balanced_dataset.py --config configs/dataset_hr_v3_balanced.json'}
    write_json(stage / 'data_dictionary.json', dictionary)
    from validate_hr_balanced_dataset import audit
    audit_report = audit(stage, source)
    write_json(stage / 'hr_augmentation_report.json', audit_report)
    if audit_report['status'] != 'PASS':
        raise ValueError(audit_report)
    manifest = json.loads((stage / 'manifest.json').read_text(encoding='utf-8'))
    manifest['files'].update(hr_augmentation_report='hr_augmentation_report.json', hr_source_events='hr_source_events.jsonl',
                             hr_source_inventory='hr_source_inventory.json')
    write_json(stage / 'manifest.json', manifest)
    write_json(stage / 'artifact_hashes.json', {p.relative_to(stage).as_posix(): file_hash(p)
        for p in sorted(stage.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'})
    if target.exists():
        backup = target.with_name('_' + target.name + '_previous_' + uuid.uuid4().hex[:8])
        target.replace(backup)
    stage.replace(target)
    print(json.dumps({'status': 'complete', 'directory': str(target),
                      'hr_field_counts': audit_report['field_counts'],
                      'source_family_overlap': audit_report['cross_split_source_families']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/dataset_hr_v3_balanced.json')
    parser.add_argument('--rebuild', action='store_true', help='Validate a fresh build, retain the previous derived version, then publish')
    args = parser.parse_args()
    build(ROOT / args.config, rebuild=args.rebuild)
