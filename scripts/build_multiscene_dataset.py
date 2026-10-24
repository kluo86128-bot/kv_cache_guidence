"""Build the approved three-scene dataset without touching old experiments.

CPU-only: python scripts/build_multiscene_dataset.py
Source snapshots are pinned by SHA256 in the produced manifest. No model is called.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import json
import math
import re
import sys
from pathlib import Path

from multiscene_common import (ISOLATED_FIELDS, LABELS, ROOT, SCENES, STRONG_FIELDS, annotation,
                               file_hash, fingerprint, make_qa, natural_name,
                               normalize, question_variants, split_name, stable,
                               value_key, write_json)
from multiscene_sources import (SOURCE_URLS, load_aml, load_employees,
                                load_medical_text, load_pii_sources, load_sap,
                                load_sgd, load_synthea)


class Selector:
    def __init__(self, cfg, pools, profiles, transactions):
        self.cfg = cfg
        self.pools = pools
        self.profiles = sorted(profiles, key=lambda p: stable(p['source_ref']))
        self.transactions = transactions
        self.owners = {}
        self.used_values = collections.defaultdict(collections.Counter)
        self.cursor = collections.Counter()
        self.selected_keys = set()
        self.rejections = collections.Counter()
        self.transaction_values = collections.defaultdict(set)
        self.profile_groups = collections.defaultdict(list)
        for p in self.profiles:
            self.profile_groups[(p['age'], p['gender'])].append(p)

    def aliases(self, record):
        aliases = set(record['related_source_keys'])
        aliases.add('text:' + fingerprint(normalize(record['native_text'])))
        for f, value in record['fields'].items():
            if f in ISOLATED_FIELDS:
                namespace = 'account' if f in {'sender_account', 'recipient_account'} else f
                aliases.add(namespace + ':' + value_key(f, value))
        return aliases

    def allowed(self, aliases, split):
        return all(self.owners.get(a, split) == split for a in aliases)

    def pool_item(self, field, split, pending):
        items = self.pools.get(field, [])
        # Circular scans allow a row skipped for one split to remain usable in another.
        start = self.cursor[field]
        for offset in range(len(items)):
            p = items[(start + offset) % len(items)]
            key = value_key(field, p['value'])
            if self.used_values[field][key]:
                continue
            aliases = {p['source_ref'], field + ':' + key}
            if self.allowed(aliases | pending, split):
                self.cursor[field] = (start + offset + 1) % len(items)
                return p
        return None

    def profile(self, text, split, pending, metadata=None):
        m = re.search(r'(?i)\b(\d{1,3})(?:[- ]year[- ]old| years? old)\b', text)
        gender = 'F' if re.search(r'(?i)\b(?:female|woman)\b', text[:700]) else 'M' if re.search(r'(?i)\b(?:male|man)\b', text[:700]) else None
        metadata = metadata or {}
        stated_age = metadata.get('subject_age') if metadata.get('subject_age') is not None else int(m.group(1)) if m else None
        gender = metadata.get('subject_gender') or gender
        if stated_age is not None:
            age = stated_age + 2026 - metadata.get('age_reference_year', 2026)
            choices = []
            for diff in (0, -1, 1):
                for sex in ([gender] if gender else ['F', 'M']):
                    choices.extend(self.profile_groups.get((age + diff, sex), []))
        else:
            choices = self.profiles
        for p in choices:
            if p['source_ref'] in self.selected_keys or self.used_values['profile'][p['source_ref']]:
                continue
            aliases = {p['source_ref']}
            for f, v in p['values'].items():
                if f in ISOLATED_FIELDS:
                    aliases.add(f + ':' + value_key(f, v))
            if self.allowed(aliases | pending, split):
                return p
        return None

    def birthday(self, record, split, pending):
        meta = record['source_metadata']
        age = meta.get('subject_age')
        expected_year = meta.get('age_reference_year', 2026) - age if age is not None else None
        for donor in self.pools.get('birth_date', []):
            value = donor['value']
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value) or not '1850-01-01' <= value <= '2026-10-16':
                continue
            if expected_year is not None and int(value[:4]) not in {expected_year, expected_year - 1}:
                continue
            if self.used_values['birth_date'][value_key('birth_date', value)]:
                continue
            aliases = {donor['source_ref'], 'birth_date:' + value_key('birth_date', value)}
            if self.allowed(aliases | pending, split):
                return donor
        return None

    def transaction(self, split, pending):
        start = self.cursor['transaction']
        for offset in range(len(self.transactions)):
            event = self.transactions[(start + offset) % len(self.transactions)]
            vals = event['values']
            if event['source_ref'] in self.selected_keys:
                continue
            if any(value_key(f, vals[f]) in self.transaction_values[f]
                   for f in ('sender_account', 'recipient_account', 'amount_paid', 'amount_received', 'timestamp')):
                continue
            aliases = {event['source_ref'], 'account:' + value_key('sender_account', vals['sender_account']),
                       'account:' + value_key('recipient_account', vals['recipient_account'])}
            if self.allowed(aliases | pending, split):
                self.cursor['transaction'] = (start + offset + 1) % len(self.transactions)
                return event
        return None

    def add_values(self, record, values, source_ref, split, *, source_row=None, components=None,
                   source_evidence=None, block_kind='identity'):
        missing = [(f, str(v)) for f, v in values.items() if v and f not in record['fields']]
        if not missing:
            return
        kind = record['record_form']
        person = 'patient' if record['scene'] == 'medical' else 'employee' if record['scene'] == 'hr' else 'customer'
        title = 'supplemental synthetic transfer' if block_kind == 'transaction' else 'supplemental synthetic ' + person + ' details'
        if kind == 'dialogue':
            speaker = 'PATIENT' if record['scene'] == 'medical' else 'USER' if record['source'] == 'SGD' else 'CUSTOMER'
            prefix = '\n\n' + speaker + ': For the ' + title + ', '
            pieces = [LABELS[f] + ' is ' + value + '.' for f, value in missing]
            block = prefix + ' '.join(pieces)
        elif kind == 'narrative':
            block = '\n\nFor the ' + title + ', ' + ' '.join(LABELS[f] + ' is ' + value + '.' for f, value in missing)
        else:
            block = '\n\n' + title.title() + '\n' + '\n'.join(LABELS[f] + ': ' + value for f, value in missing)
        begin = len(record['private_prefix'])
        record['private_prefix'] += block
        for f, value in missing:
            start = record['private_prefix'].find(value, begin)
            a = annotation(record['private_prefix'], value, source_ref, start=start,
                           locator='the ' + title, origin='cross_source_synthetic_augmentation',
                           source_row=source_row, source_evidence=(source_evidence or {}).get(f, value),
                           derived=f in {'sender_account', 'recipient_account', 'amount_paid', 'amount_received'}
                                   or (f == 'name' and components is not None))
            record['fields'][f] = value
            record['field_annotations'][f] = a
        record['related_source_keys'].append(source_ref)
        record['augmentation_blocks'].append({'kind': block_kind, 'span': [begin, len(record['private_prefix'])],
                                               'source_ref': source_ref,
                                               'association_is_synthetic': True})
        if components and any(f == 'name' for f, _ in missing):
            record['_source_components'] = components

    def proposal(self, cand, split):
        record = {'schema_version': self.cfg['schema_version'], 'scene': cand['scene'],
                  'record_form': cand['record_form'], 'split': split, 'is_synthetic': True,
                  'source': cand['source'], 'source_case_id': cand['source_uid'],
                  'native_text': cand['text'], 'private_prefix': cand['text'],
                  'fields': {f: a['value'] for f, a in cand['fields'].items()},
                  'field_annotations': copy.deepcopy(cand['fields']),
                  'related_source_keys': list(cand['related_source_keys']),
                  'augmentation_blocks': [], 'source_metadata': cand['metadata'],
                  '_source_components': cand.get('source_name_fields')}
        # Personal-name targets must be natural strings, not usernames or numbered names.
        if 'name' in record['fields'] and not natural_name(record['fields']['name']):
            record['fields'].pop('name')
            record['field_annotations'].pop('name')
        pending = self.aliases(record)
        if not self.allowed(pending, split):
            return None, 'native_cross_split_collision'
        if record['source_case_id'] in self.selected_keys or ('text:' + fingerprint(normalize(record['native_text']))) in self.selected_keys:
            return None, 'source_case_or_text_already_selected'
        if record['record_form'] == 'structured':
            prefer_unique = ('name', 'birth_date', 'hire_date', 'salary') if record['scene'] == 'hr' else ('name', 'birth_date')
            if any(f in record['fields'] and self.used_values[f][value_key(f, record['fields'][f])] for f in prefer_unique):
                return None, 'avoidable_high_cardinality_repeat'
        if record['scene'] == 'medical' and record['record_form'] != 'structured':
            profile = self.profile(record['native_text'], split, pending, record['source_metadata'])
            if profile:
                self.add_values(record, profile['values'], profile['source_ref'], split,
                                source_row=profile['source_row'], components=profile['source_name_fields'])
                record['_profile_ref'] = profile['source_ref']
                pending = self.aliases(record)
            if 'birth_date' not in record['fields']:
                donor = self.birthday(record, split, pending)
                if donor:
                    self.add_values(record, {'birth_date': donor['value']}, donor['source_ref'], split)
                    pending = self.aliases(record)
        if record['scene'] == 'finance' and record['record_form'] != 'structured':
            event = self.transaction(split, pending)
            if not event:
                return None, 'no_independent_transaction_available'
            self.add_values(record, event['values'], event['source_ref'], split,
                            source_row=event['source_row'], block_kind='transaction')
            record['_transaction'] = event
            pending = self.aliases(record)
        required = ('name', 'identity_number', 'phone', 'address') if record['scene'] == 'finance' else \
                   ('name', 'identity_number', 'phone', 'address', 'email') if record['scene'] == 'hr' else \
                   ('name', 'address', 'identity_number')
        for f in required:
            if f in record['fields']:
                continue
            donor = self.pool_item(f, split, pending)
            if donor is None:
                if f == 'name':
                    return None, 'no_name_donor_available'
                continue
            self.add_values(record, {f: donor['value']}, donor['source_ref'], split,
                            components=donor.get('source_name_fields'), source_evidence={f: donor['source_evidence']})
            pending = self.aliases(record)
        if not self.allowed(pending, split):
            return None, 'augmentation_cross_split_collision'
        if 'name' not in record['fields']:
            return None, 'missing_name'
        record['name_annotation'] = split_name(record['fields']['name'], record.pop('_source_components'))
        record['source_name_fields'] = record['name_annotation']['components']
        record['native_target_count'] = sum(a['origin'] == 'native' for a in record['field_annotations'].values())
        record['sample_kind'] = 'cross_source_augmented' if record['augmentation_blocks'] else 'source_preserving'
        record['source_metadata']['syntheticity_note'] = 'Public synthetic/scenario/test-data sources; donor association is constructed and does not identify a real shared person.'
        record['_aliases'] = list(pending)
        if cand['source'] == 'IBM-AML':
            record['_transaction'] = cand['metadata']['transaction']
        make_qa(record)
        return record, None

    def commit(self, record):
        split = record['split']
        for a in record.pop('_aliases'):
            if self.owners.get(a, split) != split:
                raise AssertionError('Cross-split alias collision')
            self.owners[a] = split
        self.selected_keys.update(record['related_source_keys'])
        self.selected_keys.add('text:' + fingerprint(normalize(record['native_text'])))
        for f, v in record['fields'].items():
            self.used_values[f][value_key(f, v)] += 1
        if profile := record.pop('_profile_ref', None):
            self.used_values['profile'][profile] += 1
        if event := record.pop('_transaction', None):
            for f in ('sender_account', 'recipient_account', 'amount_paid', 'amount_received', 'timestamp'):
                self.transaction_values[f].add(value_key(f, event['values'][f]))
            self.selected_keys.add(event['source_ref'])
        return record


def aml_candidates(events):
    result = []
    for event in events:
        text = '\n'.join(LABELS[f] + ': ' + v for f, v in event['values'].items())
        fields = {f: annotation(text, v, event['source_ref'], locator='the structured transfer record',
                                 source_row=event['source_row'], derived=f != 'timestamp' and f != 'payment_format')
                  for f, v in event['values'].items()}
        result.append({'key': event['source_ref'], 'source': 'IBM-AML', 'source_uid': event['source_ref'],
                       'scene': 'finance', 'record_form': 'structured', 'text': text, 'fields': fields,
                       'source_name_fields': None, 'related_source_keys': [event['source_ref']],
                       'metadata': {'transaction': event, 'syntheticity': 'IBM AML synthetic transfer'}})
    return result


def read_cache(path):
    return [json.loads(line) for line in path.open(encoding='utf-8')]


def write_cache(path, records):
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for r in records:
            stream.write(json.dumps(r, ensure_ascii=False) + '\n')


def load_inputs(cfg, work):
    raw = (ROOT / cfg['source_cache']).resolve()
    version_path = work / 'cache_versions.json'
    versions = json.loads(version_path.read_text(encoding='utf-8')) if version_path.exists() else {}
    if (work / 'native_candidates.jsonl').exists() and (work / 'pii_pools.json').exists() and versions.get('native') == 'clinical_extractive_v2':
        native = read_cache(work / 'native_candidates.jsonl')
        pools = json.loads((work / 'pii_pools.json').read_text(encoding='utf-8'))
    else:
        native, pools = load_pii_sources(raw)
        native.extend(load_medical_text(raw))
        native.extend(load_sap(raw))
        native.extend(load_sgd(raw))
        write_cache(work / 'native_candidates.jsonl', native)
        write_json(work / 'pii_pools.json', pools)
        versions['native'] = 'clinical_extractive_v2'
        write_json(version_path, versions)
    if (work / 'synthea_candidates.jsonl').exists():
        medical = read_cache(work / 'synthea_candidates.jsonl')
        profiles = read_cache(work / 'synthea_profiles.jsonl')
    else:
        csv_dir = ROOT / cfg['synthea_source']
        if not all((csv_dir / (table + '.csv')).exists() for table in
                   ('patients', 'conditions', 'encounters', 'medications', 'procedures', 'observations', 'allergies')):
            raise RuntimeError('Full Synthea source is not complete yet')
        medical, profiles = load_synthea(csv_dir, raw)
        write_cache(work / 'synthea_candidates.jsonl', medical)
        write_cache(work / 'synthea_profiles.jsonl', profiles)
    if (work / 'employee_candidates.jsonl').exists():
        hr = read_cache(work / 'employee_candidates.jsonl')
    else:
        hr = load_employees(raw)
        write_cache(work / 'employee_candidates.jsonl', hr)
    if (work / 'aml_events.jsonl').exists():
        transactions = read_cache(work / 'aml_events.jsonl')
    else:
        transactions = load_aml(raw)
        write_cache(work / 'aml_events.jsonl', transactions)
    all_candidates = native + medical + hr + aml_candidates(transactions[:25000])
    return all_candidates, pools, profiles, transactions


def split_targets(n):
    # Largest-remainder allocation with the approved 5:1:5 ratio.
    weights = {'train': 5, 'validation': 1, 'test': 5}
    sizes = {s: n * w // 11 for s, w in weights.items()}
    left = n - sum(sizes.values())
    order = sorted(weights, key=lambda s: (-(n * weights[s] % 11), list(weights).index(s)))
    for s in order[:left]:
        sizes[s] += 1
    return sizes


def select_all(cfg, candidates, pools, profiles, transactions, work):
    selector = Selector(cfg, pools, profiles, transactions)
    cells = collections.defaultdict(list)
    for r in candidates:
        if r['record_form'] in cfg['scenes'].get(r['scene'], []):
            cells[(r['scene'], r['record_form'])].append(r)
    inventory = {}
    for key, rs in cells.items():
        inventory['/'.join(key)] = {'candidates': len(rs), 'source_counts': dict(collections.Counter(r['source'] for r in rs))}
    write_json(work / 'candidate_inventory.json', inventory)
    output = []
    # All verified financial dialogues first, then structured source records, then native texts.
    order = [('hr', 'narrative'), ('finance', 'dialogue'), ('medical', 'structured'), ('finance', 'structured'), ('hr', 'structured'),
             ('medical', 'mixed'), ('medical', 'dialogue'),
             ('medical', 'narrative'), ('finance', 'narrative'), ('finance', 'mixed')]
    for scene, kind in order:
        rs = cells[(scene, kind)]
        quotas = split_targets(len(rs)) if (scene, kind) == ('finance', 'dialogue') else dict(cfg['ordinary_split_sizes'])
        remaining = dict(quotas)
        desired = sum(quotas.values())
        # Source ordering prioritizes real personal requests and original clinical prose.
        source_priority = {'SAP': 0, 'MTS': 0, 'MedSynth': 1, 'Synthea': 0, 'Employees': 0, 'IBM-AML': 0,
                           'Gretel-PII': 2, 'Nemotron': 3, 'Gretel-Finance': 4, 'SGD': 0}
        frequency = collections.Counter((f, value_key(f, a['value'])) for r in rs for f, a in r['fields'].items())
        def ranking(r):
            diversity = sum(1 / frequency[(f, value_key(f, a['value']))] for f, a in r['fields'].items())
            return (source_priority.get(r['source'], 5), -diversity, stable(r['key'], cfg['seed']))
        rs = sorted(rs, key=ranking)
        accepted = []
        for idx, cand in enumerate(rs):
            if sum(remaining.values()) == 0:
                break
            options = [s for s in remaining if remaining[s]]
            # Proportional deficit scheduling fills all three exact quotas deterministically.
            options.sort(key=lambda s: (-remaining[s] / quotas[s], stable(cand['key'] + s, cfg['seed'])))
            # Native aliases can already belong to a split; try that compatible split as well.
            success = False
            for split in options:
                record, why = selector.proposal(cand, split)
                if record is None:
                    selector.rejections[(scene, kind, why)] += 1
                    continue
                record = selector.commit(record)
                accepted.append(record)
                remaining[split] -= 1
                success = True
                break
            if len(accepted) and len(accepted) % 200 == 0 and success:
                print(json.dumps({'stage': 'select', 'scene': scene, 'form': kind,
                                  'accepted': len(accepted), 'target': desired, 'remaining': remaining}), flush=True)
        status = {'scene': scene, 'form': kind, 'target': quotas,
                  'actual': dict(collections.Counter(r['split'] for r in accepted)),
                  'remaining': remaining, 'candidate_rows_scanned_at_most': len(rs)}
        write_json(work / ('selection_' + scene + '_' + kind + '.json'), status)
        if any(remaining.values()):
            # Persist a concrete diagnostic instead of silently padding or duplicating records.
            write_json(work / 'selection_failure.json', {'status': 'incomplete', **status,
                         'rejections': {'/'.join(k): v for k, v in selector.rejections.items()}})
            raise RuntimeError(f'Insufficient eligible records for {scene}/{kind}: {remaining}')
        output.extend(accepted)
        print(json.dumps({'stage': 'cell_complete', **status}), flush=True)
    write_json(work / 'selection_rejections.json', {'/'.join(k): v for k, v in selector.rejections.items()})
    return output


def emit(cfg, records, work):
    publish = work / 'artifacts'
    publish.mkdir(exist_ok=True)
    split_counters = collections.Counter()
    source_rows = {}
    records.sort(key=lambda r: (r['scene'], r['record_form'], r['split'], stable(r['source_case_id'], cfg['seed'])))
    parent = list(range(len(records)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    alias_owner = {}
    for i, r in enumerate(records):
        aliases = set(r['related_source_keys'])
        for f, value in r['fields'].items():
            if f in ISOLATED_FIELDS:
                ns = 'account' if f in {'sender_account', 'recipient_account'} else f
                aliases.add(ns + ':' + value_key(f, value))
        for alias in aliases:
            if alias in alias_owner:
                a, b = find(i), find(alias_owner[alias])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                alias_owner[alias] = i
    for index, r in enumerate(records):
        key = (r['scene'], r['record_form'], r['split'])
        split_counters[key] += 1
        r['record_id'] = '_'.join(key) + '_' + str(split_counters[key]).zfill(6)
        r['dataset_id'] = cfg['dataset_id']
        # Group shared primary cases, donors and identifying values transitively.
        r['group_id'] = fingerprint(records[find(index)]['source_case_id'])[:24]
        r['native_text_sha256'] = fingerprint(r['native_text'])
        source_rows[r['source_case_id']] = {'source_ref': r['source_case_id'], 'source': r['source'],
                                           'text': r['native_text'], 'metadata': r['source_metadata']}
        # The original text is already an exact prefix of private_prefix; avoid storing it twice per record.
        r['native_text_length'] = len(r.pop('native_text'))
        if 'transaction' in r['source_metadata']:
            r['source_metadata'].pop('transaction')
    for scene, kinds in cfg['scenes'].items():
        for kind in kinds:
            cell_dir = publish / scene / kind
            cell_dir.mkdir(parents=True, exist_ok=True)
            for split in ('train', 'validation', 'test'):
                selected = [r for r in records if r['scene'] == scene and r['record_form'] == kind and r['split'] == split]
                write_cache(cell_dir / (split + '.jsonl'), selected)
    for split in ('train', 'validation', 'test'):
        write_cache(publish / (split + '.jsonl'), [r for r in records if r['split'] == split])
    write_cache(publish / 'selected_native_sources.jsonl', source_rows.values())
    write_json(publish / 'question_templates.json', question_variants())
    write_json(publish / 'build_config.json', cfg)
    write_json(publish / 'name_annotation_review.json', {
        'source_components': sum(r['name_annotation']['origin'] == 'source_components' for r in records),
        'added_rule_annotation': sum(r['name_annotation']['origin'] == 'added_rule_annotation' for r in records),
        'rule_annotations_are_not_asserted_true_cultural_name_structure': True,
        'review_is_automatic_not_a_completed_manual_review': True,
        'records_requiring_semantic_review': [r['record_id'] for r in records if r['name_annotation']['needs_human_review']],
    })
    # Human-readable previews are data artifacts (JSON/CSV), never Markdown reports.
    examples = []
    for scene, kinds in cfg['scenes'].items():
        for kind in kinds:
            train_rows = [r for r in records if r['scene'] == scene and r['record_form'] == kind and r['split'] == 'train']
            first_by_source = {}
            for r in train_rows:
                first_by_source.setdefault(r['source'], r)
            chosen = list(first_by_source.values())[:3]
            chosen_ids = {r['record_id'] for r in chosen}
            chosen.extend(r for r in train_rows if r['record_id'] not in chosen_ids)
            examples.extend(chosen[:3])
    write_cache(publish / 'preview_30.jsonl', examples)
    with (publish / 'preview_qa.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['record_id', 'scene', 'record_form', 'split', 'protocol', 'field', 'question', 'answer'])
        writer.writeheader()
        for r in examples:
            for protocol in ('training_qa', 'evaluation_qa'):
                for q in r[protocol]:
                    writer.writerow({**{k: r[k] for k in ('record_id', 'scene', 'record_form', 'split')},
                                     'protocol': protocol, **{k: q[k] for k in ('field', 'question', 'answer')}})
    raw = (ROOT / cfg['source_cache']).resolve()
    paths = sorted(raw.glob('native_*.parquet')) + sorted(raw.glob('gretel_*.parquet'))
    paths += [raw / n for n in ('native_medsynth.csv', 'native_sgd.zip', 'native_sap_hr.zip', 'HI-Small_Trans.csv')]
    paths += sorted(raw.glob('native_MTS-Dialog*.csv')) + sorted(raw.glob('load_*.dump'))
    paths += sorted((ROOT / cfg['synthea_source']).glob('*.csv'))
    manifest = {'dataset_id': cfg['dataset_id'], 'schema_version': cfg['schema_version'],
                'prepared_date': cfg['prepared_date'], 'seed': cfg['seed'], 'status': 'built_pending_independent_validation',
                'records': len(records), 'split_sizes': dict(collections.Counter(r['split'] for r in records)),
                'cells': {'/'.join(key): n for key, n in split_counters.items()},
                'target_fields': SCENES, 'source_urls': SOURCE_URLS,
                'source_files': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': file_hash(p)} for p in paths],
                'reproduction_code_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): file_hash(p) for p in
                    [ROOT / 'scripts' / name for name in ('build_multiscene_dataset.py', 'multiscene_common.py',
                                                          'multiscene_sources.py', 'validate_multiscene_dataset.py')]},
                'generator': {'population': 10000, 'seed': 2026102001, 'clinician_seed': 2026102002,
                              'reference_date': '20261016', 'end_date': '20261016',
                              'name_numbers': False, 'jar_sha256': file_hash(ROOT / 'tools/synthea-with-dependencies.jar')},
                'limitations': ['Candidate selection and extractive annotation are rule-based, not a completed manual clinical audit.',
                               'Added name segments are reversible source-order supervision, not asserted cultural given/family names.',
                               'Cross-source identities/transactions are constructed associations, never real-world record linkage.',
                               'Different representations are not automatically independent people; source and donor cases are isolated across splits.',
                               'Only verified SGD banking/payment dialogues are included; unreviewed retail support logs are excluded.',
                               'Old medical_kv four-field loader cannot consume this schema without a separate approved code update.'],
                'files': {'train': 'train.jsonl', 'validation': 'validation.jsonl', 'test': 'test.jsonl',
                          'preview': 'preview_30.jsonl', 'preview_qa': 'preview_qa.csv'}}
    write_json(publish / 'manifest.json', manifest)
    return publish


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/dataset_multiscene_v1.json')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    final = ROOT / cfg['output_directory']
    if final.exists():
        raise FileExistsError('Final dataset exists; refusing to overwrite: ' + str(final))
    work = ROOT / cfg['work_directory']
    work.mkdir(parents=True, exist_ok=True)
    candidates, pools, profiles, transactions = load_inputs(cfg, work)
    print(json.dumps({'stage': 'inputs_loaded', 'candidate_records': len(candidates),
                      'pii_pools': {f: len(v) for f, v in pools.items()}, 'patient_profiles': len(profiles),
                      'transactions': len(transactions)}), flush=True)
    if args.prepare_only:
        return
    records = select_all(cfg, candidates, pools, profiles, transactions, work)
    artifacts = emit(cfg, records, work)
    from validate_multiscene_dataset import validate
    report = validate(artifacts, write=True)
    if report['status'] != 'PASS':
        raise RuntimeError('Independent validation failed; staging artifacts retained for repair')
    artifacts.rename(final)
    print(json.dumps({'status': 'complete', 'output': str(final), 'records': len(records),
                      'validation': 'PASS'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
