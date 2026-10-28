"""Audit balanced HR artifacts against the frozen parent and saved event donors."""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
from pathlib import Path

from multiscene_common import ROOT, file_hash, fingerprint, normalize


def audit(folder, parent_folder):
    folder, parent_folder = Path(folder), Path(parent_folder)
    cfg = json.loads((folder / 'hr_augmentation_config.json').read_text(encoding='utf-8'))
    keep = tuple(cfg['hr_fields'])
    groups = {'leave': ('leave_start', 'leave_duration', 'leave_reason'),
              'complaint': ('complaint_target', 'complaint_content')}
    forbidden = re.compile(r'(?i)supplemental synthetic|HR-MultiWOZ|WorkplaceDispute|source_ref|association_is_synthetic|https?://(?:huggingface\.co|github\.com/SAP|arxiv\.org)')
    donors = [json.loads(line) for line in (folder / 'hr_source_events.jsonl').open(encoding='utf-8')]
    donor_map = {d['source_id']: d for d in donors}
    errors, owner, family_owner = [], {}, {}
    for d in donors:
        for key, mapping in ((d['source_id'], owner), (d['event_group'], family_owner)):
            if key in mapping and mapping[key] != d['assigned_split']:
                errors.append('Source event/family crosses splits: ' + key)
            mapping[key] = d['assigned_split']
    # Independent normalization of source content: names and numbers do not
    # make identical/near-identical scenarios independent source events.
    def normalized_scenario(d):
        content = d['values'].get('leave_reason', d['values'].get('complaint_content', ''))
        names = ([p.split(' (')[0] for p in d['source_row']['participants']]
                 if d['source'] == 'WorkplaceDispute' else [d['values'].get('complaint_target', '')])
        for name in names:
            for word in name.split():
                if len(word) > 2:
                    content = re.sub(r'\b' + re.escape(word) + r'\b', 'PERSON', content, flags=re.I)
        content = re.sub(r'\d+', 'NUMBER', normalize(content))
        words = re.findall(r'[a-z]+', content)
        return {tuple(words[i:i + 3]) for i in range(max(1, len(words) - 2))}
    signatures = [normalized_scenario(d) for d in donors]
    near_cross = []
    for i, d in enumerate(donors):
        for j in range(i):
            other = donors[j]
            if d['kind'] != other['kind'] or d['assigned_split'] == other['assigned_split']:
                continue
            denominator = len(signatures[i] | signatures[j])
            similarity = len(signatures[i] & signatures[j]) / denominator if denominator else 1.
            if similarity >= cfg['near_duplicate_jaccard']:
                near_cross.append([d['source_id'], other['source_id'], similarity])
    if near_cross:
        errors.append('Near-duplicate source scenarios crossed splits')
    counts, forms, uses, answers = {}, {}, {}, {}
    preserved_other = collections.Counter()
    changes = collections.Counter()
    inherited_incompatible_dates = []
    added = collections.Counter()
    group_owners = {}
    for split in ('train', 'validation', 'test'):
        before = [json.loads(line) for line in (parent_folder / (split + '.jsonl')).open(encoding='utf-8')]
        after = [json.loads(line) for line in (folder / (split + '.jsonl')).open(encoding='utf-8')]
        if [r['record_id'] for r in before] != [r['record_id'] for r in after]:
            errors.append('Parent record IDs/order/count changed: ' + split)
        counts[split], forms[split], uses[split] = collections.Counter(), collections.defaultdict(collections.Counter), collections.Counter()
        answers[split] = collections.defaultdict(collections.Counter)
        for old, new in zip(before, after):
            if old['scene'] != 'hr':
                if old != new:
                    errors.append('Other scene changed: ' + old['record_id'])
                preserved_other[old['scene']] += 1
                continue
            rid = new['record_id']
            # Keep inherited source values but never add adult employee events
            # to records whose known dates rule them out.
            incompatible = False
            for field, days in (('birth_date', 18 * 365), ('hire_date', 1)):
                try:
                    day = dt.date.fromisoformat(old['fields'].get(field, ''))
                except ValueError:
                    continue
                incompatible |= day + dt.timedelta(days=days) > dt.date(2026, 10, 16)
            if incompatible:
                inherited_incompatible_dates.append(rid)
                if new['hr_derivation']['events']:
                    errors.append('New event on incompatible inherited dates: ' + rid)
            if set(new['fields']) - set(keep) or new['split'] != old['split'] or new['record_form'] != old['record_form']:
                errors.append('HR scope/split/form changed: ' + rid)
            if new['private_prefix'][:new['native_text_length']] != old['private_prefix'][:old['native_text_length']]:
                errors.append('Cleaned native input changed: ' + rid)
            if new['parent_group_id'] != old['group_id']:
                errors.append('Parent group audit changed: ' + rid)
            for f, v in old['fields'].items():
                if f in keep and new['fields'].get(f) != v:
                    errors.append('Existing retained gold changed: ' + rid + '/' + f)
            if forbidden.search(new['private_prefix']) or any(forbidden.search(q['question'])
                    for protocol in ('qa', 'training_qa', 'evaluation_qa') for q in new[protocol]):
                errors.append('Source marker in model input/question: ' + rid)
            if new['group_id'] in group_owners and group_owners[new['group_id']] != split:
                errors.append('Connected group crossed splits')
            group_owners[new['group_id']] = split
            for kind, fields in groups.items():
                present = set(fields) & set(new['fields'])
                if present and present != set(fields):
                    errors.append('Incomplete business event: ' + rid + '/' + kind)
            for f, v in new['fields'].items():
                counts[split][f] += 1
                forms[split][new['record_form']][f] += 1
                answers[split][f][normalize(v)] += 1
                a = new['field_annotations'][f]
                if f not in old['fields']:
                    added[f] += 1
                    donor = donor_map.get(a['source_ref'])
                    if (not donor or a.get('source_assigned_split') != split or donor['assigned_split'] != split
                            or a.get('source_event_group') != donor['event_group']):
                        errors.append('Donor provenance/split mismatch: ' + rid + '/' + f)
            for event in new['hr_derivation']['events']:
                if owner.get(event['source_id']) != split or family_owner.get(event['event_group']) != split:
                    errors.append('Reused source in wrong split: ' + rid)
                uses[split][event['event_group']] += 1
            changes['changed_hr_inputs'] += new['private_prefix'] != old['private_prefix']
        for f in keep:
            n = counts[split][f]
            if n < cfg['coverage_per_field'][split]:
                errors.append('Field below minimum coverage: ' + split + '/' + f)
            if f in set(sum((list(v) for v in groups.values()), [])) and n != cfg['coverage_per_field'][split]:
                errors.append('Expanded field quota mismatch: ' + split + '/' + f)
        for form, fc in forms[split].items():
            for f in sum((list(v) for v in groups.values()), []):
                if fc[f] != cfg['coverage_per_field'][split] // 2:
                    errors.append('Expanded field form quota mismatch: ' + split + '/' + form + '/' + f)
    overlap = {}
    for f in keep:
        common = set(answers['train'][f]) & set(answers['test'][f])
        overlap[f] = {'train_distinct_values': len(answers['train'][f]), 'test_distinct_values': len(answers['test'][f]),
                      'train_test_shared_values': len(common),
                      'test_records_with_train_seen_value': sum(answers['test'][f][v] for v in common),
                      'interpretation': 'Common answer reuse is measured, not automatically a source-event leak.'}
    hashes = json.loads((parent_folder / 'artifact_hashes.json').read_text(encoding='utf-8'))
    for name, sha in hashes.items():
        if file_hash(parent_folder / name) != sha:
            errors.append('Frozen parent artifact changed: ' + name)
    return {'status': 'PASS' if not errors else 'FAIL', 'error_count': len(errors), 'errors': errors[:100],
            'field_counts': counts, 'field_counts_by_form': forms,
            'new_field_instances': dict(added), 'input_changes': dict(changes),
            'inherited_records_excluded_from_new_adult_events': inherited_incompatible_dates,
            'unchanged_other_scene_records': dict(preserved_other),
            'hr_connected_groups_by_split': dict(collections.Counter(group_owners.values())),
            'cross_split_source_families': 0 if not errors else 'see_errors',
            'cross_split_near_source_scenarios': near_cross[:30],
            'near_duplicate_rule': 'Normalized source content, person/digit masking, trigram Jaccard >= ' + str(cfg['near_duplicate_jaccard']),
            'source_reuse': {s: {'used_families': len(u), 'maximum_family_uses': max(u.values(), default=0)} for s, u in uses.items()},
            'answer_overlap': overlap, 'manual_semantic_review_complete': False,
            'scope': 'Record preservation, literal evidence via general validator, quotas, provenance-free input/questions and declared source-family isolation. Not proof against every semantic paraphrase or a full manual event-consistency review.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='data/dataset_multiscene_v3_hr_balanced')
    parser.add_argument('--parent', default='data/dataset_multiscene_v2_reduced')
    args = parser.parse_args()
    from validate_multiscene_dataset import validate
    base = validate(ROOT / args.directory)
    result = audit(ROOT / args.directory, ROOT / args.parent)
    print(json.dumps({'general_validation': base['status'], **result}, ensure_ascii=False))
    if base['status'] != 'PASS' or result['status'] != 'PASS':
        raise SystemExit(1)
