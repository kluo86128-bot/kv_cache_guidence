"""Select and split source-preserving Synthea patient profiles; never edit values."""
import collections
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import unicodedata
from pathlib import Path

from preview_source import QUESTIONS, ROOT
sys.path.insert(0, str(ROOT))
from medical_kv.name_policy import NAME_FORMAT, natural_patient

sys.path.insert(0, str(ROOT / 'tools/python'))
import tokenizers
from tokenizers import Tokenizer


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stable(value, seed):
    return hashlib.sha256(f'{seed}|{value}'.encode()).hexdigest()


def normalized(value):
    # Normalization is used only to detect grouping collisions, never to alter inputs.
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def full_name(p):
    return ' '.join(p[k] for k in ('FIRST', 'MIDDLE', 'LAST') if p[k] != '')


def load_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


class Flow:
    def __init__(self, n):
        self.graph = [[] for _ in range(n)]

    def add(self, u, v, cap):
        forward = [v, len(self.graph[v]), cap]
        reverse = [u, len(self.graph[u]), 0]
        self.graph[u].append(forward)
        self.graph[v].append(reverse)
        return forward

    def maximum(self, source, sink):
        total = 0
        while True:
            level = [-1] * len(self.graph)
            level[source] = 0
            queue = collections.deque([source])
            while queue:
                u = queue.popleft()
                for v, _, cap in self.graph[u]:
                    if cap and level[v] < 0:
                        level[v] = level[u] + 1
                        queue.append(v)
            if level[sink] < 0:
                return total
            pos = [0] * len(self.graph)

            def send(u, amount):
                if u == sink:
                    return amount
                while pos[u] < len(self.graph[u]):
                    edge = self.graph[u][pos[u]]
                    v, rev, cap = edge
                    if cap and level[v] == level[u] + 1:
                        pushed = send(v, min(amount, cap))
                        if pushed:
                            edge[2] -= pushed
                            self.graph[v][rev][2] += pushed
                            return pushed
                    pos[u] += 1
                return 0

            while pushed := send(source, 1000000):
                total += pushed


def describe(values):
    values = sorted(values)
    return {'min': values[0], 'mean': round(statistics.mean(values), 3),
            'median': statistics.median(values),
            'p95': values[math.ceil(.95 * len(values)) - 1], 'max': values[-1]}


def main():
    cfg_path = ROOT / 'configs/dataset_v1.json'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    seed = cfg['seed']
    csv_dir, output = ROOT / cfg['source_csv_directory'], ROOT / cfg['output_directory']
    files = {k: csv_dir / (k + '.csv') for k in ('patients', 'encounters', 'conditions')}
    if not all(p.is_file() and p.stat().st_size for p in files.values()):
        raise RuntimeError('Source generation is not complete: missing or empty CSV')
    patients_rows, encounter_rows, condition_rows = (load_csv(files[k]) for k in files)
    patients_rows = [natural_patient(r) for r in patients_rows]
    patients = {r['Id']: r for r in patients_rows}
    encounters = {r['Id']: r for r in encounter_rows}
    assert len(patients) == len(patients_rows)
    assert digest(ROOT / 'tools/synthea-with-dependencies.jar') == cfg['generation']['jar_sha256']

    preview = [json.loads(x) for x in (ROOT / 'data/preview_100/samples_100.jsonl').read_text(encoding='utf-8').splitlines()]
    preview_pids = {r['provenance']['source_patient_id'] for r in preview}
    preview_names = {normalized(r['fields']['name']) for r in preview}
    preview_addresses = {normalized(r['fields']['address']) for r in preview}

    groups = UnionFind(patients)
    seen_name, seen_address = {}, {}
    for pid, p in patients.items():
        for value, seen in ((normalized(full_name(p)), seen_name), (normalized(p['ADDRESS']), seen_address)):
            if value:
                if value in seen:
                    groups.union(pid, seen[value])
                else:
                    seen[value] = pid
    excluded_groups = set()
    for pid, p in patients.items():
        if (pid in preview_pids or normalized(full_name(p)) in preview_names
                or normalized(p['ADDRESS']) in preview_addresses):
            excluded_groups.add(groups.find(pid))

    latest = collections.defaultdict(dict)
    condition_filter = collections.Counter()
    for c in condition_rows:
        pid, eid, label = c['PATIENT'], c['ENCOUNTER'], c['DESCRIPTION']
        if pid not in patients or eid not in encounters or encounters[eid]['PATIENT'] != pid:
            condition_filter['invalid_source_join'] += 1
            continue
        if not label.endswith('(disorder)'):
            condition_filter['not_disorder_label'] += 1
            continue
        old = latest[pid].get(label)
        if old is None or (encounters[eid]['START'], eid, c['CODE']) > (
                encounters[old['ENCOUNTER']]['START'], old['ENCOUNTER'], old['CODE']):
            latest[pid][label] = c

    eligible_groups = collections.defaultdict(list)
    rejected = collections.Counter()
    exclusion_rows = []
    for pid, p in patients.items():
        group = groups.find(pid)
        reason = None
        if group in excluded_groups:
            reason = 'preview_identity_or_address_overlap'
        elif not all(p[k] for k in ('FIRST', 'LAST', 'BIRTHDATE', 'ADDRESS')):
            reason = 'missing_target_value'
        elif not latest[pid]:
            reason = 'no_joined_disorder_diagnosis'
        if reason:
            rejected[reason] += 1
            exclusion_rows.append({'source_patient_id': pid, 'reason': reason})
        else:
            eligible_groups[group].append(pid)

    representatives = []
    for group, members in sorted(eligible_groups.items()):
        ordered = sorted(members, key=lambda pid: stable(pid, seed))
        representatives.append(ordered[0])
        for pid in ordered[1:]:
            rejected['additional_same_identity_or_address_group_member'] += 1
            exclusion_rows.append({'source_patient_id': pid, 'reason': 'additional_same_identity_or_address_group_member', 'representative': ordered[0]})

    k = cfg['diagnosis_categories']
    split_sizes = cfg['split_sizes']
    split_quotas = {s: [n // k + (i < n % k) for i in range(k)] for s, n in split_sizes.items()}
    maximum_quota = max(sum(split_quotas[s][i] for s in split_sizes) for i in range(k))
    availability = collections.Counter(label for pid in representatives for label in latest[pid])
    ranked_labels = sorted(availability, key=lambda label: (-availability[label], label))
    labels = sorted([label for label in ranked_labels if availability[label] >= maximum_quota][:k])
    if len(labels) < k:
        raise RuntimeError(f'Only {len(labels)} diagnoses have {maximum_quota} distinct eligible patients; generate more candidates.')

    pids = sorted(representatives, key=lambda pid: stable(pid, seed + 1))
    n = len(pids)
    source, sink = n + k, n + k + 1
    flow = Flow(n + k + 2)
    tracked = []
    for i, pid in enumerate(pids):
        flow.add(source, i, 1)
        for j, label in enumerate(labels):
            if label in latest[pid]:
                tracked.append((pid, j, flow.add(i, n + j, 1)))
    for j in range(k):
        flow.add(n + j, sink, sum(split_quotas[s][j] for s in split_sizes))
    matched = flow.maximum(source, sink)
    target = sum(split_sizes.values())
    if matched != target:
        raise RuntimeError(f'Cannot jointly satisfy diagnosis quotas: matched {matched}/{target}; generate additional candidates.')
    by_label = collections.defaultdict(list)
    for pid, j, edge in tracked:
        if edge[2] == 0:
            by_label[labels[j]].append(pid)

    assigned = collections.defaultdict(list)
    for j, label in enumerate(labels):
        # Weighted interleaving along sorted birth dates spreads all splits over
        # the same age range while exactly satisfying every diagnosis quota.
        ordered = sorted(by_label[label], key=lambda pid: (patients[pid]['BIRTHDATE'], stable(pid, seed + 2)))
        quota = {s: split_quotas[s][j] for s in split_sizes}
        filled = collections.Counter()
        total = len(ordered)
        for i, pid in enumerate(ordered):
            allowed = [s for s in split_sizes if filled[s] < quota[s]]
            split = max(allowed, key=lambda s: ((i + 1) * quota[s] / total - filled[s], stable(f'{label}|{s}', seed + 3)))
            filled[split] += 1
            assigned[split].append((pid, latest[pid][label]))
        assert dict(filled) == quota

    tokenizer_path = ROOT / cfg['tokenizer_file']
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    token_count = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    source_hashes = {k: digest(p) for k, p in files.items()}
    output.mkdir(parents=True, exist_ok=True)
    source_selected = {k: [] for k in files}
    all_records, split_reports = [], {}
    selected_ids = set()
    for split in split_sizes:
        entries = assigned[split]
        random.Random(f'{seed}|{split}').shuffle(entries)
        records = []
        for i, (pid, c) in enumerate(entries, 1):
            p, e = patients[pid], encounters[c['ENCOUNTER']]
            names = {key: p[key] for key in ('FIRST', 'MIDDLE', 'LAST')}
            fields = {'name': full_name(p), 'birth_date': p['BIRTHDATE'], 'address': p['ADDRESS'], 'diagnosis': c['DESCRIPTION']}
            prefix = '\n'.join([f'patients.{key}: {value}' for key, value in names.items()] + [
                f"patients.BIRTHDATE: {fields['birth_date']}", f"patients.ADDRESS: {fields['address']}",
                f"conditions.DESCRIPTION: {fields['diagnosis']}"])
            record = {
                'record_id': f'{split}_{i:06d}', 'split': split, 'is_synthetic': True,
                'dataset_id': cfg['dataset_id'], 'schema_version': cfg['schema_version'], 'name_format': NAME_FORMAT,
                'group_id': stable(groups.find(pid), seed + 4)[:24],
                'fields': fields, 'source_name_fields': names, 'private_prefix': prefix,
                'qa': [{'field': key, 'question': QUESTIONS[key], 'answer': value} for key, value in fields.items()],
                'token_lengths': {'private_prefix': token_count(prefix),
                                  'answers': {key: token_count(value) for key, value in fields.items()}},
                'provenance': {
                    'dataset': 'Synthea locally generated source CSV, build d9d07a6',
                    'generation_config': 'configs/synthea_candidates_5000.properties',
                    'generator_jar_sha256': cfg['generation']['jar_sha256'],
                    'source_csv_sha256': source_hashes,
                    'source_patient_id': pid, 'source_encounter_id': e['Id'],
                    'source_condition_code': c['CODE'], 'source_condition_system': c['SYSTEM'],
                    'source_condition_description': c['DESCRIPTION'],
                    'source_condition_start': c['START'], 'source_encounter_start': e['START'],
                    'name_format': NAME_FORMAT,
                    'transformations': 'Remove terminal numeric name suffixes; join nonempty FIRST/MIDDLE/LAST with one space. Birth date, address and diagnosis copied unchanged; English labels/questions added.'
                },
            }
            assert pid not in selected_ids
            selected_ids.add(pid)
            assert all(q['answer'] == fields[q['field']] for q in record['qa'])
            records.append(record)
            for table, row in (('patients', p), ('encounters', e), ('conditions', c)):
                source_selected[table].append(row)
        assert len(records) == split_sizes[split]
        (output / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
        split_reports[split] = {
            'records': len(records), 'qa_pairs': len(records) * 4,
            'diagnosis_counts': dict(sorted(collections.Counter(r['fields']['diagnosis'] for r in records).items())),
            'birth_decade_counts': dict(sorted(collections.Counter(r['fields']['birth_date'][:3] + '0s' for r in records).items())),
            'private_prefix_tokens': describe([r['token_lengths']['private_prefix'] for r in records]),
            'answer_tokens': {key: describe([r['token_lengths']['answers'][key] for r in records]) for key in cfg['target_fields']},
            'distinct_values': {key: len({r['fields'][key] for r in records}) for key in cfg['target_fields']},
        }
        all_records.extend(records)
    assert len(all_records) == target
    split_sets = {s: [r for r in all_records if r['split'] == s] for s in split_sizes}
    for i, a in enumerate(split_sizes):
        for b in list(split_sizes)[i + 1:]:
            for key in ('name', 'address'):
                assert not ({normalized(r['fields'][key]) for r in split_sets[a]} & {normalized(r['fields'][key]) for r in split_sets[b]})
    (output / 'all_records.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in all_records), encoding='utf-8')
    write_json(output / 'source_rows_selected.json', source_selected)
    write_json(output / 'diagnosis_labels.json', labels)
    write_json(output / 'candidate_exclusions.json', exclusion_rows)
    write_json(output / 'statistics.json', split_reports)
    not_selected = len(representatives) - len(selected_ids)
    manifest = {
        'dataset_id': cfg['dataset_id'], 'schema_version': cfg['schema_version'], 'prepared_date': '2026-10-16',
        'records': target, 'independent_patients': len(selected_ids), 'qa_pairs': target * 4,
        'split_sizes': split_sizes, 'target_fields': cfg['target_fields'],
        'source_values_modified': True, 'translations_applied': False, 'name_format': NAME_FORMAT,
        'name_transformation': 'Remove terminal ASCII digits from name components; no other field normalization.',
        'source_row_counts': {k: len(v) for k, v in zip(files, (patients_rows, encounter_rows, condition_rows))},
        'source_csv_sha256': source_hashes, 'generation': cfg['generation'],
        'candidate_pipeline': {'excluded_patients_by_reason': dict(rejected),
                               'condition_rows_filtered': dict(condition_filter),
                               'eligible_identity_address_groups': len(eligible_groups),
                               'eligible_representatives': len(representatives),
                               'not_selected_after_balanced_quotas': not_selected},
        'diagnosis_selection': {'count': k, 'rule': 'Top 20 disorder descriptions by independent representative-patient availability, each >= maximum per-label total quota; lexical tie-break; no model outcomes used.',
                                'labels': labels, 'candidate_patient_availability': {label: availability[label] for label in labels}},
        'grouping': 'NFKC+casefold+whitespace-normalized name or ADDRESS connected components; keep one eligible representative per component. Values themselves unchanged.',
        'splitting': 'Exact diagnosis quotas via maximum bipartite matching; age-aware weighted interleaving of sorted birth dates within each diagnosis; seeded final shuffling.',
        'preview_exclusion': 'All four splits exclude source patient IDs and normalized full names/addresses overlapping displayed 100-patient preview.',
        'birth_date_overlap_allowed': True, 'diagnosis_label_overlap_required': True,
        'tokenizer': {'label': cfg['tokenizer_label'], 'file': cfg['tokenizer_file'],
                      'sha256': digest(tokenizer_path), 'library': 'tokenizers', 'version': tokenizers.__version__,
                      'add_special_tokens': False, 'scope': 'plain private_prefix and answer strings; no chat template, U or C included'},
        'runtime': {'python': platform.python_version(), 'system': platform.platform()},
        'build_config_sha256': digest(cfg_path), 'build_script_sha256': digest(Path(__file__)),
        'input_scope': 'Only private_prefix enters encoding; QA labels for supervision only; provenance/record_id/group_id excluded from model inputs.',
        'status': 'Built; independent validation report must be checked before use.',
    }
    write_json(output / 'manifest.json', manifest)
    artifact_hashes = {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != 'artifact_hashes.json'}
    write_json(output / 'artifact_hashes.json', artifact_hashes)
    print(json.dumps({'records': target, 'splits': split_sizes, 'source_patients': len(patients),
                      'eligible_representatives': len(representatives), 'diagnoses': labels}, indent=2))


if __name__ == '__main__':
    main()
