"""Deterministic HR event donors, isolation and source-free model text.

All values added here describe constructed employee/event associations. Source
IDs, licenses, source wording and derivation rules stay outside model inputs.
"""
from __future__ import annotations

import ast
import collections
import copy
import csv
import datetime as dt
import json
import random
import re
from pathlib import Path

from multiscene_common import annotation, fingerprint, make_qa, normalize, stable

LEAVE = ('leave_start', 'leave_duration', 'leave_reason')
COMPLAINT = ('complaint_target', 'complaint_content')
GROUPS = {'leave': LEAVE, 'complaint': COMPLAINT}
SPLITS = ('train', 'validation', 'test')
MARKERS = re.compile(r'(?i)supplemental synthetic|HR-MultiWOZ|WorkplaceDispute|source_ref|association_is_synthetic|https?://(?:huggingface\.co|github\.com/SAP|arxiv\.org)')
CONDUCT = re.compile(r'(?i)\b(?:yell\w*|insult\w*|shout\w*|harass\w*|discriminat\w*|'
    r'touch\w*|threat\w*|bully\w*|stole|steal\w*|falsif\w*|forg\w*|'
    r'withhold\w*|withheld|unpaid|overtime|paycheck\w*|wages?|racial|racist|sexist|'
    r'comment\w*|jokes?|promot\w*|favorit\w*|favourit\w*|retaliat\w*|'
    r'confidential|disclos\w*|leak\w*|unsafe|safety|grop\w*|sabotag\w*|'
    r'timecards?|timesheets?|timecards?|brib\w*|embezzl\w*|fraud\w*)\b')


def complaint_subject(description, involved):
    """Prefer an explicit actor in the incident; reject a cast containing self."""
    match = re.search(r'(?i)\b(my (?:manager|supervisor|colleague|co-worker|coworker|boss|teacher|student|director)|the student)\b', description)
    if match:
        return match.group()
    involved = re.sub(r'(?i)^(?:I|me|myself)\s+and\s+', '', involved.strip())
    involved = re.sub(r'(?i)\s+and\s+(?:I|me|myself)$', '', involved)
    if re.search(r'(?i)\b(?:I|me|myself)\b', involved):
        return None
    return involved if involved and len(involved) <= 150 else None


def read_donors(cache):
    """Read inert CSV/JSON, never pickle or executable publisher code."""
    cache = Path(cache)
    donors, rejected = [], collections.Counter()
    for row in csv.DictReader((cache / 'hr_multiwoz.csv').open(encoding='utf-8-sig', newline='')):
        service = ast.literal_eval(row['service'])[0]
        if service not in ('time_off_report', 'harassment_report'):
            continue
        states = {k: str(v).strip() for state in ast.literal_eval(row['state']) for k, v in state.items()}
        utterances = ast.literal_eval(row['utterance'])
        text = '\n'.join(utterances)
        key = 'HR-MultiWOZ:' + row['dialogue_id']
        if service == 'time_off_report':
            values = {'leave_start': states.get('start_date', ''),
                      'leave_duration': states.get('total_days_off', ''),
                      'leave_reason': states.get('reason_for_time_off', '')}
            if not all(values.values()) or not values['leave_duration'].isdigit():
                rejected['incomplete_leave'] += 1
                continue
            # The dataset state sometimes loses units or exact wording. This is
            # a documented derivation, not a claim of unchanged source extraction.
            values['leave_duration'] += ' days'
            if len(values['leave_reason']) < 8 or len(values['leave_reason']) > 220:
                rejected['invalid_leave_reason'] += 1
                continue
            kind = 'leave'
        else:
            values = {'complaint_target': states.get('individuals_involved', ''),
                      'complaint_content': states.get('description_of_incident', '')}
            if (not all(values.values()) or not 15 <= len(values['complaint_content']) <= 300
                    or len(values['complaint_target']) > 150):
                rejected['incomplete_hr_complaint'] += 1
                continue
            target = complaint_subject(values['complaint_content'], values['complaint_target'])
            if not target:
                rejected['ambiguous_hr_complaint_cast'] += 1
                continue
            values['complaint_target'] = target
            kind = 'complaint'
        donors.append({'source_id': key, 'source': 'HR-MultiWOZ', 'kind': kind,
                       'values': values, 'source_text': text, 'source_row': states,
                       'derivation': 'HR slot scenario reused as a fictional event; individuals_involved is the cast, not automatically an accused person.'})
    for line in (cache / 'workplace_dispute.jsonl').open(encoding='utf-8'):
        row = json.loads(line)
        participants = row['participants']
        cast = []
        for person in participants:
            match = re.fullmatch(r'(.+?)\s*\((.+)\)', person)
            if match:
                cast.append((match[1].strip(), match[2].strip()))
        # Use a participant as the cast for a NEW fictional complaint, never
        # infer that every source participant is guilty or is the complainant.
        if len(cast) < 2:
            rejected['missing_cast'] += 1
            continue
        candidates = []
        for turn in row['dialogue']:
            original_text = turn['text'].strip()
            text = original_text
            if not 25 <= len(text) <= 300:
                continue
            # Require a statement about a concrete conduct, not only an insult
            # or rhetorical question. Keep wording from the source utterance.
            statements = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text)
                          if s.strip() and not s.strip().endswith('?') and CONDUCT.search(s)]
            text = ' '.join(statements)
            if not 25 <= len(text) <= 300:
                continue
            for name, role in cast:
                if name != turn['speaker'] and re.search(r'\b' + re.escape(name.split()[0]) + r'\b', original_text):
                    candidates.append((name, role, text))
        if not candidates:
            rejected['no_explicit_addressee'] += 1
            continue
        name, role, utterance = candidates[0]
        donors.append({'source_id': 'WorkplaceDispute:' + row['transcript_id'],
                       'source': 'WorkplaceDispute', 'kind': 'complaint',
                       'values': {'complaint_target': name + ' (' + role + ')',
                                  'complaint_content': utterance},
                       'topic': row['conflict_type'], 'source_row': row,
                       'source_text': '\n'.join(t['speaker'] + ': ' + t['text'] for t in row['dialogue']),
                       'derivation': 'Named addressee and source wording reused in an explicitly constructed employee complaint; no real employee linkage.'})
    return donors, dict(rejected)


def signature_text(donor):
    text = donor['values'].get('leave_reason', donor['values'].get('complaint_content', ''))
    # Remove cast/dates/digits so renamed/re-dated copies remain one family.
    row = donor['source_row']
    names = []
    if donor['source'] == 'WorkplaceDispute':
        names = [p.split(' (')[0] for p in row['participants']]
    else:
        names = [donor['values'].get('complaint_target', '')]
    for name in names:
        for part in name.split():
            if len(part) > 2:
                text = re.sub(r'\b' + re.escape(part) + r'\b', 'PERSON', text, flags=re.I)
    text = re.sub(r'\d+', 'NUMBER', normalize(text))
    return ' '.join(re.findall(r'[a-z]+', text))


def shingles(text):
    words = text.split()
    return {tuple(words[i:i + 3]) for i in range(max(1, len(words) - 2))}


def group_and_partition(donors, seed, threshold=.72):
    """Union exact and near-duplicate source scenarios BEFORE assigning owners."""
    parent = list(range(len(donors)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(i, j):
        parent[find(i)] = find(j)
    texts = [signature_text(d) for d in donors]
    grams = [shingles(t) for t in texts]
    for i, donor in enumerate(donors):
        for j in range(i):
            if donor['kind'] != donors[j]['kind']:
                continue
            union = grams[i] | grams[j]
            similarity = len(grams[i] & grams[j]) / len(union) if union else 1.
            if texts[i] == texts[j] or similarity >= threshold:
                join(i, j)
    groups = collections.defaultdict(list)
    for i, donor in enumerate(donors):
        groups[find(i)].append(donor)
    assignments = []
    # Stratification keeps both publishers represented while connected families
    # remain indivisible. Counts need not equal 32/7/32 after deduplication.
    strata = collections.defaultdict(list)
    for members in groups.values():
        group_id = 'hr_event:' + fingerprint('|'.join(sorted(d['source_id'] for d in members)))[:24]
        strata[(members[0]['kind'], '|'.join(sorted({d['source'] for d in members})))].append((group_id, members))
    for entries in strata.values():
        entries.sort(key=lambda pair: stable(pair[0], seed))
        if len(entries) < 3:
            raise ValueError('Too few independent donor families for three splits')
        nval = max(1, round(len(entries) / 11))
        ntrain = (len(entries) - nval) // 2
        for index, (group, members) in enumerate(entries):
            split = 'train' if index < ntrain else 'validation' if index < ntrain + nval else 'test'
            for donor in members:
                assignments.append(dict(donor, event_group=group, assigned_split=split))
    return assignments


def select_records(rows, fields, target, seed, label):
    """Preserve every native partial event; fill each form to half the quota."""
    chosen = set()
    for form in ('structured', 'narrative'):
        candidates = [r for r in rows if r['record_form'] == form]
        existing = [r for r in candidates if set(fields) & set(r['fields'])]
        quota = target // 2
        if len(existing) > quota or quota > len(candidates) or target % 2:
            raise ValueError('Cannot satisfy coverage/form quota without dropping existing events')
        chosen.update(r['record_id'] for r in existing)
        remaining = [r for r in candidates if r['record_id'] not in chosen and adult_event_eligible(r)]
        if len(remaining) < quota - len(existing):
            raise ValueError('Too few age/date-compatible records for event quota')
        remaining.sort(key=lambda r: stable(label + '|' + r['record_id'], seed))
        chosen.update(r['record_id'] for r in remaining[:quota - len(existing)])
    return chosen


def adult_event_eligible(record):
    """Do not invent adult employee events for inherited recent birth dates."""
    for field, days in (('birth_date', 18 * 365), ('hire_date', 1)):
        try:
            day = dt.date.fromisoformat(record['fields'].get(field, ''))
        except ValueError:
            continue
        if day + dt.timedelta(days=days) > dt.date(2026, 10, 16):
            return False
    return True


class DonorCycle:
    """Balance reuse at family level, then rotate members within a family."""
    def __init__(self, donors, seed, split, kind):
        groups = collections.defaultdict(list)
        for d in donors:
            if d['assigned_split'] == split and d['kind'] == kind:
                groups[d['event_group']].append(d)
        self.groups = sorted(groups, key=lambda s: stable(f'{split}|{kind}|{s}', seed))
        self.members = {g: sorted(v, key=lambda d: d['source_id']) for g, v in groups.items()}
        self.index = 0
        if not self.groups:
            raise ValueError('Empty donor partition')

    def next(self):
        g = self.groups[self.index % len(self.groups)]
        cycle = self.index // len(self.groups)
        donor = self.members[g][cycle % len(self.members[g])]
        self.index += 1
        return donor


def clean_parent_record(original, keep):
    """Keep cleaned native input and original values; re-render only added text."""
    r = copy.deepcopy(original)
    r['fields'] = {f: v for f, v in r['fields'].items() if f in keep}
    r['field_annotations'] = {f: a for f, a in r['field_annotations'].items() if f in keep}
    r['hr_derivation'] = {'parent_private_prefix_sha256': fingerprint(original['private_prefix']),
                          'removed_target_fields': sorted(set(original['fields']) - set(keep)),
                          'native_input_unchanged': True, 'events': []}
    r['private_prefix'] = original['private_prefix'][:original['native_text_length']]
    r['augmentation_blocks'] = []
    for f, a in r['field_annotations'].items():
        if a['origin'] == 'native':
            continue
        old = copy.deepcopy(a)
        add_block(r, {f: r['fields'][f]}, a['source_ref'], 'employee_details', annotations={f: old})
    return r


def add_block(r, values, source_ref, kind, *, annotations=None, donor=None, recipe=None):
    """Place provenance-free business facts in P and metadata outside P/Q."""
    titles = {'employee_details': 'Employee details', 'leave': 'Leave request details',
              'complaint': 'Employee complaint'}
    labels = {'leave_start': 'Leave start', 'leave_duration': 'Leave duration',
              'leave_reason': 'Leave reason', 'complaint_target': 'Person complained about',
              'complaint_content': 'Conduct complained about'}
    from multiscene_common import LABELS
    title = titles[kind]
    begin = len(r['private_prefix'])
    parts = []
    for field, value in values.items():
        label = labels.get(field, LABELS[field])
        parts.append(label + ': ' + value)
    if r['record_form'] == 'narrative':
        # These are sentences documenting specific business facts, not source tags.
        block = '\n\n' + title + '. ' + ' '.join(p if p.endswith(('.', '!', '?')) else p + '.' for p in parts)
    else:
        block = '\n\n' + title + '\n' + '\n'.join(parts)
    r['private_prefix'] += block
    cursor = begin
    for field, value in values.items():
        # Match the label+value to avoid duplicate substrings inside earlier facts.
        label = labels.get(field, LABELS[field]) + ': '
        start = r['private_prefix'].index(label + value, cursor) + len(label)
        cursor = start + len(value)
        if annotations:
            a = annotations[field]
            a.update(evidence_span=[start, cursor], locator='the ' + title.lower() + ' section')
        else:
            a = annotation(r['private_prefix'], value, source_ref, start=start,
                           locator='the ' + title.lower() + ' section',
                           origin='cross_source_synthetic_augmentation', derived=True,
                           source_row=donor['source_row'],
                           source_evidence=donor['values'].get(field, ''),)
            a['derivation'] = recipe
            a['source_event_group'] = donor['event_group']
            a['source_assigned_split'] = donor['assigned_split']
        r['fields'][field] = value
        r['field_annotations'][field] = a
    if source_ref not in r['related_source_keys']:
        r['related_source_keys'].append(source_ref)
    block_meta = {'kind': kind, 'span': [begin, len(r['private_prefix'])],
                  'source_ref': source_ref, 'association_is_synthetic': True}
    if donor:
        block_meta.update(event_group=donor['event_group'], source_id=donor['source_id'],
                          source_assigned_split=donor['assigned_split'])
    r['augmentation_blocks'].append(block_meta)


def event_values(r, kind, donor, seed):
    rng = random.Random(stable(r['record_id'] + '|' + kind, seed))
    values = {}
    recipe = {'method': 'source_scenario_and_deterministic_business_record_v1',
              'source_id': donor['source_id'], 'event_group': donor['event_group'],
              'association_is_synthetic': True, 'original_values_preserved': True}
    if kind == 'leave':
        lower = dt.date(2018, 1, 1)
        for field in ('hire_date', 'birth_date'):
            try:
                day = dt.date.fromisoformat(r['fields'].get(field, ''))
            except ValueError:
                continue
            lower = max(lower, day + dt.timedelta(days=1 if field == 'hire_date' else 18 * 365))
        upper = dt.date(2026, 10, 16)
        if lower > upper:
            raise ValueError('No valid leave date after existing hire/birth dates')
        day = lower + dt.timedelta(days=rng.randint(0, (upper - lower).days))
        formats = ('iso', 'slash', 'written')
        style = formats[rng.randrange(len(formats))]
        values['leave_start'] = (day.isoformat() if style == 'iso' else day.strftime('%d/%m/%Y')
                                 if style == 'slash' else day.strftime('%B %d, %Y'))
        # Existing partial events determine the other facts when present; no
        # end date is inferred from ambiguous original durations such as months.
        duration = donor['values']['leave_duration']
        values['leave_duration'] = duration
        values['leave_reason'] = donor['values']['leave_reason']
        recipe.update(date_policy='Calendar date after recorded hire date and adulthood, at or before 2026-10-16',
                      duration_policy='Source scenario duration; unchanged native partial fields take precedence')
    else:
        if donor['source'] == 'HR-MultiWOZ':
            # A NEW constructed complaint uses the source involved-person string
            # as its explicit target; the original role is never claimed verified.
            values = dict(donor['values'])
        else:
            values = dict(donor['values'])
        recipe['role_policy'] = 'Source cast/addressee is assigned the explicit target role in this new fictional complaint; not a validated accusation about the original dialogue.'
    return {f: v for f, v in values.items() if f not in r['fields']}, recipe


def augment_rows(rows, donors, keep, coverage, seed):
    rows = [clean_parent_record(r, keep) for r in rows]
    for split in SPLITS:
        subset = [r for r in rows if r['split'] == split]
        for kind, fields in GROUPS.items():
            chosen = select_records(subset, fields, coverage[split], seed, kind)
            cycle = DonorCycle(donors, seed, split, kind)
            for r in sorted(subset, key=lambda r: stable(kind + '|' + r['record_id'], seed)):
                if r['record_id'] not in chosen or all(f in r['fields'] for f in fields):
                    continue
                donor = cycle.next()
                values, recipe = event_values(r, kind, donor, seed)
                ref = donor['source_id']
                add_block(r, values, ref, kind, donor=donor, recipe=recipe)
                r['hr_derivation']['events'].append(recipe)
        for r in subset:
            r['native_target_count'] = sum(a['origin'] == 'native' for a in r['field_annotations'].values())
            r['sample_kind'] = 'cross_source_augmented' if r['augmentation_blocks'] else 'source_preserving'
            make_qa(r)
            if MARKERS.search(r['private_prefix']) or any(MARKERS.search(q['question']) for q in r['training_qa'] + r['evaluation_qa']):
                raise ValueError('Provenance marker entered model input: ' + r['record_id'])
    return rows


def connect_augmented_groups(rows):
    """Keep parent groups and all reused source events together for development views."""
    owner, parent = {}, list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, r in enumerate(rows):
        r['parent_group_id'] = r['group_id']
        aliases = ['parent:' + r['group_id']] + r['related_source_keys']
        aliases += [e['event_group'] for e in r['hr_derivation']['events']]
        for alias in aliases:
            if alias in owner:
                j = owner[alias]
                if rows[j]['split'] != r['split']:
                    raise ValueError('Source family crossed split boundaries')
                parent[find(i)] = find(j)
            owner[alias] = i
    members = collections.defaultdict(list)
    for i, r in enumerate(rows):
        members[find(i)].append(r['record_id'])
    ids = {i: fingerprint('|'.join(sorted(rs)))[:24] for i, rs in members.items()}
    for i, r in enumerate(rows):
        r['group_id'] = ids[find(i)]
