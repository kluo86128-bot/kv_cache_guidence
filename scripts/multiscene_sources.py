"""Pinned public-source readers. Extractive labels always point into retained text."""
from __future__ import annotations

import collections
import csv
import io
import json
import re
import sys
import zipfile
from pathlib import Path

from multiscene_common import (ROOT, SCENES, annotation, fingerprint, form_of,
                               natural_name, normalize, parse_entities, stable)

LABEL_MAP = {
    'name': 'name', 'full_name': 'name', 'person': 'name',
    'date_of_birth': 'birth_date', 'dob': 'birth_date',
    'address': 'address', 'street_address': 'address',
    'ssn': 'identity_number', 'national_id': 'identity_number',
    'phone_number': 'phone', 'phone': 'phone', 'email': 'email', 'email_address': 'email',
    'occupation': 'title', 'job_title': 'title', 'employment_status': 'employment_status',
    'health_plan_beneficiary_number': 'benefit_member_number',
    'account_number': 'recorded_account_number', 'iban': 'recorded_account_number',
    'bban': 'recorded_account_number', 'credit_debit_card': 'card_number',
    'credit_card_number': 'card_number',
}
MEDICAL_DOMAINS = {'Healthcare', 'Health', 'Healthcare Providers', 'healthcare', 'healthcare-administration'}
FINANCE_DOMAINS = {'Finance', 'Banking', 'Credit', 'Brokerage', 'Investment', 'Mortgage',
                   'banking', 'finance', 'financial-services', 'digital-payments'}
HR_TYPES = {
    'employment contract', 'employmentcontract', 'employeecontract', 'payroll record',
    'employment verification letter', 'employment verification', 'employment application',
    'employee onboarding form', 'employee onboarding security documentation',
    'personnel security file', 'employee benefits enrollment', 'performance review',
    'termination or resignation notice', "worker's compensation claim",
    'pensionbenefitstatement',
}
SOURCE_URLS = {
    'Nemotron': 'https://huggingface.co/datasets/nvidia/Nemotron-PII/tree/b70ffaf5ff39e079776134c5bf4381f00a9fd1ed',
    'Gretel-PII': 'https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1/tree/e06eb1499ca8d54470f085021cd8e54f9efac7fd',
    'Gretel-Finance': 'https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual',
    'MedSynth': 'https://huggingface.co/datasets/Ahmad0067/MedSynth/tree/ea5ae2c59217fdda5bf8754102bc260260d0d825',
    'MTS': 'https://github.com/abachaa/MTS-Dialog',
    'SGD': 'https://github.com/google-research-datasets/dstc8-schema-guided-dialogue',
    'SAP': 'https://github.com/SAP/hr-request-data-set',
    'Employees': 'https://github.com/datacharmer/test_db',
    'IBM-AML': 'https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml',
    'Synthea': 'https://github.com/synthetichealth/synthea',
}


def connection(raw):
    deps = raw.parent / 'deps'
    if deps.is_dir():
        sys.path.insert(0, str(deps))
    import duckdb
    c = duckdb.connect()
    c.execute("set memory_limit='3GB'")
    c.execute('set threads=4')
    return c


def entity_spans(text, entities):
    spans = []
    for e in parse_entities(entities):
        if not isinstance(e, dict):
            continue
        value = e.get('entity', e.get('text'))
        a, b = e.get('start'), e.get('end')
        if value is None and isinstance(a, int) and isinstance(b, int):
            value = text[a:b]
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()
        if value not in text or '[' in value or ']' in value or 'placeholder' in value.casefold():
            continue
        if isinstance(a, int) and isinstance(b, int) and value in text[a:b]:
            a += text[a:b].find(value)
        else:
            a = text.find(value)
        for label in e.get('types') or [e.get('label')]:
            if label:
                spans.append({'start': a, 'end': a + len(value), 'value': value, 'label': label.casefold()})
    return sorted(spans, key=lambda s: (s['start'], s['end'], s['label']))


def extract_entities(text, entities, ref):
    spans = entity_spans(text, entities)
    fields = {}
    supplied = None
    for s in spans:
        field = LABEL_MAP.get(s['label'])
        if not field or field in fields:
            continue
        v = s['value']
        if field == 'name' and not natural_name(v):
            continue
        if field == 'phone' and not 10 <= len(re.sub(r'\D', '', v)) <= 15:
            continue
        if field == 'identity_number' and not 6 <= len(re.sub(r'[^\w]', '', v)) <= 20:
            continue
        if field == 'card_number' and not 13 <= len(re.sub(r'\D', '', v)) <= 19:
            continue
        if normalize(v) in {'unknown', 'not specified', 'n/a', 'not provided', 'not available'}:
            continue
        fields[field] = annotation(text, v, ref, start=s['start'],
                                   locator='the first explicitly recorded ' + field.replace('_', '-') + ' entry')
    # Adjacent, labelled components have an explicit correspondence in the text.
    for i, a in enumerate(spans):
        if a['label'] != 'first_name':
            continue
        for b in spans[i + 1:]:
            if b['label'] != 'last_name' or b['start'] < a['end']:
                continue
            gap = text[a['end']:b['start']]
            if not re.fullmatch(r'\s+', gap):
                continue
            value = text[a['start']:b['end']]
            if not natural_name(value) or ' '.join([a['value'], b['value']]) != value:
                continue
            if 'name' not in fields or a['start'] <= fields['name']['evidence_span'][0]:
                fields['name'] = annotation(text, value, ref, start=a['start'],
                                             locator='the first explicitly written personal-name entry')
                supplied = {'FIRST': a['value'], 'MIDDLE': '', 'LAST': b['value']}
            break
        if supplied:
            break
    if 'name' not in fields:
        first = next((s for s in spans if s['label'] == 'first_name' and natural_name(s['value'])), None)
        if first:
            fields['name'] = annotation(text, first['value'], ref, start=first['start'],
                                         locator='the first explicitly recorded given-name entry')
            # A recorded partial name is segmented; absence is not asserted as a real-world fact.
    return fields, supplied, spans


def trimmed_span(text, match, group='value'):
    value = match.group(group)
    stripped = value.strip(' \t*\r')
    if not stripped or len(stripped) > 240:
        return None
    start = match.start(group) + value.find(stripped)
    return stripped, start


def header(text, labels):
    pat = r'(?im)^[ \t]*(?:[\d.*#()•-]+[ \t]*)?(?:\*\*)?(?:' + labels + r')(?:\*\*)?\s*:\s*(?:\*\*)?[ \t]*(?P<value>[^\n]+)'
    m = re.search(pat, text)
    return trimmed_span(text, m) if m else None


def add_header(fields, text, ref, field, labels):
    if field in fields:
        return
    result = header(text, labels)
    if result:
        value, start = result
        fields[field] = annotation(text, value, ref, start=start,
                                   locator='the first ' + labels.split('|')[0].replace('?', '') + ' entry')


def section_item(text, section):
    pat = r'(?im)^[ \t]*(?:\*\*)?(?:\d+[.)][ \t]*)?' + section + r'(?:\*\*)?[ \t]*:?[ \t]*(?:\*\*)?[ \t]*$'
    m = re.search(pat, text)
    if not m:
        return None
    offset = m.end()
    for line in text[offset:].splitlines(keepends=True):
        stripped = line.strip(' \t\r\n-*•')
        if stripped:
            if len(stripped) > 240:
                return None
            return stripped, offset + line.find(stripped)
        offset += len(line)
    return None


def clean_clinical_line(line, offset):
    """A contiguous answer span; remove only exterior formatting, never rewrite words."""
    m = re.match(r'^[ \t]*(?:#{1,6}[ \t]*)?(?:[-*•][ \t]+|\d+[.)][ \t]+)?(?:\*\*)?', line)
    start = m.end() if m else 0
    end = len(line.rstrip(' \t\r\n*:'))
    while start < end and line[start] in ' \t*':
        start += 1
    if start >= end:
        return None
    return line[start:end], offset + start


def diagnosis_from_assessment(text):
    """Prefer the actual assessment title/item over an explanatory 'Diagnosis: Based on...' line."""
    active = False
    offset = 0
    skip_labels = {'primary diagnosis', 'diagnosis', 'diagnoses', 'assessment',
                   'clinical assessment', 'secondary diagnosis', 'secondary diagnoses',
                   'differential diagnosis', 'differential diagnoses', 'medical reasoning'}
    for line in text.splitlines(keepends=True):
        clean = line.replace('*', '').replace('#', '').strip()
        clean = re.sub(r'^\d+[.)]\s*', '', clean).strip().rstrip(':')
        if re.fullmatch(r'(?i)assessment', clean):
            active = True
            offset += len(line)
            continue
        if not active:
            offset += len(line)
            continue
        if re.match(r'(?i)^(?:plan|treatment plan|management plan)(?:\s*:|$)', clean):
            break
        result = clean_clinical_line(line, offset)
        offset += len(line)
        if not result:
            continue
        value, start = result
        if value.casefold().rstrip(':') in skip_labels:
            continue
        m = re.match(r'(?i)(?:primary diagnosis|diagnosis|diagnoses)\s*:?\s*(?:\*\*)?\s*:\s*(?:\*\*)?\s*(.+)', value)
        if m:
            new = m.group(1).strip(' *')
            start += value.find(new)
            value = new
        if re.match(r'(?i)^(?:based on|confirmed by|medical reasoning|patient presents|clinical reasoning|differential diagnosis|secondary diagnoses)', value):
            continue
        if not 4 <= len(value) <= 240 or not re.search(r'[A-Za-z]{3}', value):
            continue
        if value.count('**'):
            # An exterior label can be stripped only when the remaining answer is contiguous.
            pos = value.find(':')
            if pos >= 0:
                new = value[pos + 1:].strip(' *')
                if not new:
                    continue
                start += value.find(new)
                value = new
        if re.match(r'(?i)^(?:based on|confirmed by)', value):
            continue
        return value, start
    return None


def first_section_excerpt(text):
    stripped = text.strip()
    if not stripped:
        return None
    if len(stripped) <= 240:
        return stripped, text.find(stripped)
    for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z])', stripped):
        if sentence.endswith(('Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Jr.', 'Sr.')):
            continue
        if 15 <= len(sentence) <= 240:
            return sentence, text.find(sentence)
    return None


def subject_demographics(text):
    m = re.search(r'(?i)\b(\d{1,3})(?:[- ]year[- ]old| years? old)\b', text[:4500])
    gender = 'F' if re.search(r'(?i)\b(?:female|woman)\b', text[:1800]) else 'M' if re.search(r'(?i)\b(?:male|man)\b', text[:1800]) else None
    date = re.search(r'(?im)^\s*(?:\*\*)?Date(?:\*\*)?\s*:\s*(\d{4})-\d{2}-\d{2}', text)
    return {'subject_age': int(m.group(1)) if m else None, 'subject_gender': gender,
            'age_reference_year': int(date.group(1)) if date else 2026}


def augment_native_fields(text, ref, scene, fields):
    if scene == 'medical':
        for f, labels in [('birth_date', r'date of birth|birth date|DOB'),
                          ('address', r'home address|residential address|street address'),
                          ('identity_number', r'SSN|social security number'),
                          ('encounter_start', r'encounter start|encounter date|visit date|date of service'),
                          ('diagnosis', r'primary diagnosis|final diagnosis|diagnosis'),
                          ('medication', r'current medications?|medications?|prescribed medications?'),
                          ('procedure', r'procedures?|surgical procedures?'),
                          ('allergy', r'allergies|allergy|known allergies')]:
            add_header(fields, text, ref, f, labels)
        for f, sec in [('diagnosis', 'Assessment'), ('medication', 'Current Medications|Medications'),
                       ('allergy', 'Allergies'), ('procedure', 'Procedures')]:
            if f not in fields and (result := section_item(text, sec)):
                value, start = result
                fields[f] = annotation(text, value, ref, start=start,
                                       locator='the first item in the ' + sec.split('|')[0] + ' section')
        if result := diagnosis_from_assessment(text):
            value, start = result
            fields['diagnosis'] = annotation(text, value, ref, start=start,
                                             locator='the first condition entry in the Assessment section')
        elif 'diagnosis' in fields and re.match(r'(?i)^(?:based on|confirmed by|medical reasoning|primary diagnosis)', fields['diagnosis']['value']):
            fields.pop('diagnosis')
        if 'observation' not in fields:
            m = re.search(r'(?i)\b(?:blood pressure|heart rate|blood glucose|HbA1c|creatinine|hemoglobin|temperature|oxygen saturation)\s*(?:is|of|:|=)\s*\d[\d./]*(?:\s*(?:mmHg|bpm|beats per minute|mg/dL|mmol/L|%|g/dL|°[CF]|degrees))?', text)
            if m:
                fields['observation'] = annotation(text, m.group(), ref, start=m.start(),
                                                   locator='the first explicit numerical observation or vital-sign result')
        if 'name' not in fields:
            m = re.search(r'\b(?:patient.{0,70}?\bnamed|patient name\s*:)\s*([A-Z][a-zÀ-ž\'-]+(?: [A-Z][a-zÀ-ž\'-]+){1,3})(?=[,.\n])', text)
            if m and natural_name(m.group(1)):
                fields['name'] = annotation(text, m.group(1), ref, start=m.start(1),
                                             locator='the explicitly named patient entry')
    if scene == 'hr':
        for f, labels in [('department', r'department|division'), ('title', r'job title|position|occupation'),
                          ('hire_date', r'hire date|date of hire|employment start date'),
                          ('benefit_plan', r'benefit plan|insurance plan|health plan name'),
                          ('benefit_coverage', r'personal coverage|employee coverage|coverage amount'),
                          ('leave_reason', r'leave reason|reason for leave'),
                          ('leave_start', r'leave start|start date of leave'),
                          ('leave_duration', r'leave duration'),
                          ('complaint_content', r'complaint details|grievance details')]:
            add_header(fields, text, ref, f, labels)
        if 'salary' not in fields:
            m = re.search(r'(?i)\b(?:current (?:annual |monthly )?salary|annual salary|monthly salary|base salary|salary)\s*(?:is|of|:|=)\s*(?P<value>(?:USD|US\$|\$|€|£)?\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:euros?|USD|dollars?|per year|per month))?)', text)
            if m:
                value, start = trimmed_span(text, m)
                fields['salary'] = annotation(text, value, ref, start=start,
                                              locator='the first explicitly stated salary amount')
    return fields


def candidate(source, uid, scene, form, text, fields, supplied=None, meta=None):
    return {'key': source + ':' + str(uid) + ':' + form,
            'source': source, 'source_uid': source + ':' + str(uid),
            'scene': scene, 'record_form': form, 'text': text,
            'fields': fields, 'source_name_fields': supplied,
            'related_source_keys': [source + ':' + str(uid)],
            'metadata': meta or {}}


def route(domain, doctype):
    if domain in {'Human Resources', 'human-resources'} or doctype.casefold() in HR_TYPES:
        return 'hr'
    if domain in MEDICAL_DOMAINS:
        return 'medical'
    if domain in FINANCE_DOMAINS:
        return 'finance'
    return None


def load_pii_sources(raw):
    c = connection(raw)
    result = []
    pools = collections.defaultdict(list)
    for source, paths, col in [
        ('Nemotron', [raw / ('native_nemotron_' + s + '.parquet') for s in ('train', 'test')], 'spans'),
        ('Gretel-PII', [raw / ('native_gretel_pii_' + s + '.parquet') for s in ('train', 'validation', 'test')], 'entities')]:
        cur = c.execute('select uid,domain,document_type,text,' + col + ' from read_parquet(?)',
                        [[str(p) for p in paths]])
        batch_no = 0
        while batch := cur.fetchmany(2000):
            batch_no += 1
            for uid, domain, dt, text, entities in batch:
                if len(text.strip()) < 30 or re.match(r"(?is)^\s*(?:I'm sorry|I am sorry|As an AI|I cannot|I'm unable)", text):
                    continue
                ref = source + ':' + uid
                fs, supplied, spans = extract_entities(text, entities, ref)
                for f in ('name', 'birth_date', 'identity_number', 'phone', 'address', 'email'):
                    if f in fs:
                        pools[f].append({'value': fs[f]['value'], 'source_ref': ref,
                                         'source_evidence': fs[f]['value'],
                                         'source_name_fields': supplied if f == 'name' else None,
                                         'source_text_hash': fingerprint(text)})
                scene = route(domain, dt)
                if scene is None:
                    continue
                kind = form_of(text, dt)
                if kind not in {'narrative', 'mixed'}:
                    continue
                fs = augment_native_fields(text, ref, scene, fs)
                fs = {f: a for f, a in fs.items() if f in SCENES[scene]}
                if not fs:
                    continue
                result.append(candidate(source, uid, scene, kind, text, fs, supplied,
                                        {'domain': domain, 'document_type': dt,
                                         'document_genre_method': 'operational_rules',
                                         'native_annotation_method': 'publisher_entities_and_conservative_literal_rules'}))
            if batch_no % 20 == 0:
                print(json.dumps({'stage': 'native_pii', 'source': source, 'batches': batch_no}), flush=True)
    fs_paths = [str(raw / 'gretel_train.parquet'), str(raw / 'gretel_test.parquet')]
    rows = c.execute("select index,document_type,generated_text,pii_spans from read_parquet(?) where language='English'", [fs_paths]).fetchall()
    excluded = {'FIX Protocol', 'MT940', 'CSV', 'SWIFT Message', 'FpML', 'EDI', 'XBRL',
                'BAI Format', 'IT support ticket', 'Bill of Lading', 'Safety Data Sheet'}
    for idx, dt, text, entities in rows:
        if dt == 'Customer support conversational log' or dt in excluded:
            # SGD supplies the verified full dialogue set. Support logs remain audit candidates.
            continue
        scene = 'hr' if dt == 'Employment Contract' else 'medical' if dt == 'Health Insurance Claim Form' else 'finance'
        kind = form_of(text, dt)
        if kind not in {'narrative', 'mixed'}:
            continue
        ref = 'Gretel-Finance:' + str(idx)
        fs, supplied, _ = extract_entities(text, entities, ref)
        fs = augment_native_fields(text, ref, scene, fs)
        fs = {f: a for f, a in fs.items() if f in SCENES[scene]}
        if fs:
            result.append(candidate('Gretel-Finance', str(idx), scene, kind, text, fs, supplied,
                                    {'document_type': dt, 'native_annotation_method': 'publisher_spans_and_literal_rules'}))
    # One original locale variant per UID and genre; no merging regional facts.
    best = {}
    for r in result:
        key = (r['source_uid'], r['record_form'])
        score = (len(r['fields']), bool(r['source_name_fields']), stable(r['text']))
        if key not in best or score > best[key][0]:
            best[key] = (score, r)
    result = [v[1] for v in best.values()]
    for f in pools:
        seen = set()
        unique = []
        for p in sorted(pools[f], key=lambda p: stable(p['source_ref'] + p['value'])):
            key = normalize(p['value'])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        pools[f] = unique
    return result, dict(pools)


def load_medical_text(raw):
    result = []
    rows = list(csv.DictReader((raw / 'native_medsynth.csv').open(encoding='utf-8-sig')))
    parent = list(range(len(rows)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for col in (' Note', 'Dialogue'):
        owners = {}
        for i, row in enumerate(rows):
            h = fingerprint(normalize(row[col]))
            if h in owners:
                a, b = find(i), find(owners[h])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                owners[h] = i
    used = set()
    for i, row in enumerate(rows):
        case = find(i)
        if case in used:
            continue
        used.add(case)
        ref = 'MedSynth:' + str(case)
        for kind, text in [('mixed', row[' Note']), ('dialogue', row['Dialogue'])]:
            fields = augment_native_fields(text, ref, 'medical', {})
            # ICD metadata can disagree with the actual text. It is never copied as an answer.
            if kind == 'mixed' and form_of(text, 'SOAP note') != 'mixed':
                kind = form_of(text, 'SOAP note')
            if kind not in {'mixed', 'narrative', 'dialogue'}:
                continue
            meta = {'native_annotation_method': 'literal_headers_sections_and_observations',
                    'publisher_icd10': row.get('ICD10'), 'publisher_icd10_desc': row.get('ICD10_desc'),
                    'publisher_icd_metadata_used_as_gold': False,
                    'paired_note_dialogue_case': ref, **subject_demographics(row[' Note'])}
            # Transfer no ICD label. A note-derived candidate is used in a dialogue only if
            # the exact condition text is actually spoken there, including any qualifiers.
            if kind == 'dialogue' and 'diagnosis' not in fields:
                diag = diagnosis_from_assessment(row[' Note'])
                if diag:
                    value = diag[0]
                    value = re.sub(r'\s*\((?:ICD[^)]*|[A-Z]\d[^)]*)\)\.?$', '', value).rstrip('.')
                    m = re.search(re.escape(value), text, flags=re.I)
                    if m and len(value) >= 4:
                        fields['diagnosis'] = annotation(text, m.group(), ref, start=m.start(),
                                                         locator='the first spoken occurrence of the assessment condition')
            if fields:
                result.append(candidate('MedSynth', case, 'medical', kind, text, fields, meta=meta))
    # Summaries and dialogues share a source case. Duplicate text is handled by the selector.
    for path in sorted(raw.glob('native_MTS-Dialog*.csv')):
        for i, row in enumerate(csv.DictReader(path.open(encoding='utf-8-sig'))):
            uid = path.name + ':' + str(i)
            ref = 'MTS:' + uid
            text = row['section_text']
            fields = augment_native_fields(text, ref, 'medical', {})
            section = row.get('section_header', '').upper()
            # These named sections explicitly identify the clinical category.
            field = {'MEDICATIONS': 'medication', 'MEDS': 'medication', 'ALLERGY': 'allergy',
                     'ALLERGIES': 'allergy', 'DIAGNOSIS': 'diagnosis', 'ASSESSMENT': 'diagnosis',
                     'GENHX': 'medical_history', 'PASTMEDICALHX': 'medical_history',
                     'FAM/SOCHX': 'family_social_history', 'CC': 'symptom', 'ROS': 'symptom',
                     'PASTSURGICAL': 'procedure', 'PROCEDURES': 'procedure', 'GYNHX': 'medical_history',
                     'OTHER_HISTORY': 'medical_history', 'PLAN': 'clinical_plan', 'DISPOSITION': 'clinical_plan',
                     'EDCOURSE': 'clinical_plan', 'EXAM': 'observation', 'IMAGING': 'observation',
                     'LABS': 'observation', 'IMMUNIZATIONS': 'medical_history'}.get(section)
            excerpt = first_section_excerpt(text)
            if field and excerpt:
                value, start = excerpt
                fields[field] = annotation(text, value, ref, start=start,
                                           locator='the first complete excerpt in the supplied ' + section + ' section')
            # Without an extractable clinical target, demographics can still be added, but this
            # candidate is not counted as natively field-labelled and is selected after labelled cases.
            result.append(candidate('MTS', uid, 'medical', 'narrative', text, fields,
                                    meta={'section_header': section, **subject_demographics(text),
                                          'native_annotation_method': 'publisher_section_and_literal_rules',
                                          'clinical_text_has_no_extractable_target': not bool(fields)}))
    return result


SAP_MAP = {'description_life_event': 'personal_event', 'reason': 'leave_reason',
           'date_start_absence': 'leave_start', 'number_of_days': 'leave_duration',
           'duration': 'leave_duration', 'complaint': 'complaint_content', 'to_who': 'complaint_target',
           'reason_of_change': 'shift_change_reason', 'date_travel': 'travel_date',
           'location': 'travel_destination', 'increase_in_percentage': 'requested_raise',
           'wage_gap': 'pay_gap', 'work_title': 'title'}


def sap_fields(text, row, ref):
    fields = augment_native_fields(text, ref, 'hr', {})
    category = row.get('label', row.get('category', '')).casefold()
    rejected = []
    for item in row.get('entities', []):
        if len(item) < 3:
            continue
        start, end, lab = item
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)):
            rejected.append(item)
            continue
        value = text[start:end].strip()
        # Publisher offsets include known shifted/truncated spans. Reject token-interior bounds.
        offset = start + len(text[start:end]) - len(text[start:end].lstrip())
        actual_end = offset + len(value)
        boundary_ok = ((offset == 0 or not (text[offset - 1].isalnum() and text[offset].isalnum())) and
                       (actual_end == len(text) or not (text[actual_end - 1].isalnum() and text[actual_end].isalnum())))
        if not boundary_ok or not value or len(value) > 240:
            rejected.append(item)
            continue
        field = SAP_MAP.get(lab)
        if lab == 'salary':
            # Explicit salary parsing handles current/requested amounts; do not label both current.
            continue
        if field and field not in fields:
            if field == 'leave_duration' and not re.search(r'(?i)\b(?:days?|weeks?|months?|hours?|years?|fortnight)\b', value):
                rejected.append(item)
                continue
            fields[field] = annotation(text, value, ref, start=offset,
                                       locator='the first explicitly stated ' + field.replace('_', '-') + ' detail')
    patterns = {
        'leave_start': r'(?i)\b(?:on|the|from|starting)\s+(?P<value>\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?)',
        'leave_duration': r'(?i)\b(?P<value>(?:a|one|two|three|\d+)\s+(?:working\s+)?(?:days?|weeks?|months?|hours?)(?:\s+off)?)\b',
        'requested_raise': r'(?i)\b(?:increase|raise).{0,25}?(?P<value>\d+(?:\.\d+)?\s*%)',
        'requested_salary': r'(?i)\b(?:to|salary of)\s+(?P<value>\d[\d,.]*\s*(?:euros?|dollars?|USD|GBP))',
    }
    for f, pat in patterns.items():
        if f not in fields and (m := re.search(pat, text)):
            fields[f] = annotation(text, m.group('value'), ref, start=m.start('value'),
                                    locator='the first explicit ' + f.replace('_', '-') + ' statement')
    # Whole sentences remain original extracts, rather than invented category-name answers.
    for m in re.finditer(r'[^.!?\n]+[.!?]?', text):
        sentence = m.group().strip()
        if not 12 <= len(sentence) <= 240:
            continue
        f = None
        if 'complaint' in category and re.search(r'(?i)\b(?:harass|discriminat|offend|unfair|colleague|manager|favour|favor|bully)', sentence):
            f = 'complaint_content'
        elif 'life event' in category and re.search(r'(?i)\b(?:illness|hospital|medical|funeral|death|marri|pregnan|sick|family|personal)', sentence):
            f = 'personal_event'
        elif 'shift' in category and re.search(r'(?i)\b(?:because|due|need|have to|reason)', sentence):
            f = 'shift_change_reason'
        if f and f not in fields:
            fields[f] = annotation(text, sentence, ref, start=text.find(sentence),
                                    locator='the first sentence describing the personal ' + f.replace('_', '-') + ' detail')
    return fields, rejected


def load_sap(raw):
    result = []
    z = zipfile.ZipFile(raw / 'native_sap_hr.zip')
    seen = set()
    for suffix in ('/survey_tickets/data.json', '/tickets_handwritten.json'):
        member = next(n for n in z.namelist() if n.endswith(suffix))
        for row in json.loads(z.read(member)):
            text = row['ticket']
            h = fingerprint(normalize(text))
            if h in seen:
                continue
            seen.add(h)
            ref = 'SAP:' + suffix + ':' + str(row['id'])
            fields, rejected = sap_fields(text, row, ref)
            if fields:
                result.append(candidate('SAP', suffix + ':' + str(row['id']), 'hr', 'narrative', text, fields,
                                        meta={'request_category': row.get('label'),
                                              'rejected_publisher_offsets': rejected,
                                              'native_annotation_method': 'checked_publisher_offsets_and_literal_rules',
                                              'syntheticity': 'human_authored_scenario_request; not asserted to be an actual employee event'}))
    return result


def load_sgd(raw):
    result = []
    z = zipfile.ZipFile(raw / 'native_sgd.zip')
    for member in sorted(z.namelist()):
        if not re.search(r'/(train|dev|test)/dialogues_\d+\.json$', member):
            continue
        for row in json.loads(z.read(member)):
            services = [s for s in row['services'] if s.startswith(('Banks_', 'Payment'))]
            if not services:
                continue
            # Separate outer source split from ID; related text hashes are grouped by selector.
            uid = member.split('/')[-2] + ':' + row['dialogue_id']
            ref = 'SGD:' + uid
            turns = row['turns']
            text = '\n'.join(t['speaker'] + ': ' + t['utterance'] for t in turns)
            native_slots = []
            offset = 0
            for turn in turns:
                for fr in turn.get('frames', []):
                    if fr.get('service') not in services:
                        continue
                    for sl in fr.get('slots', []):
                        a, b = sl.get('start'), sl.get('exclusive_end')
                        if isinstance(a, int) and isinstance(b, int) and 0 <= a < b <= len(turn['utterance']):
                            native_slots.append({'slot': sl['slot'], 'value': turn['utterance'][a:b],
                                                 'start': offset + len(turn['speaker']) + 2 + a})
                offset += len(turn['speaker']) + 2 + len(turn['utterance']) + 1
            # Preserve native financial information in its actual category; never call a name
            # an account number or a balance a payment amount.
            slot_map = {'account_balance': 'account_balance', 'balance': 'account_balance',
                        'amount': 'transfer_amount', 'transfer_amount': 'transfer_amount',
                        'recipient_name': 'recipient_name', 'receiver': 'recipient_name',
                        'account_type': 'account_type', 'transfer_time': 'transfer_time'}
            fields = {}
            for slot in sorted(native_slots, key=lambda s: s['start']):
                f = slot_map.get(slot['slot'])
                if f and f not in fields:
                    fields[f] = annotation(text, slot['value'], ref, start=slot['start'])
            result.append(candidate('SGD', uid, 'finance', 'dialogue', text, fields,
                                    meta={'services': services, 'native_slots': native_slots,
                                          'native_annotation_method': 'published_dialogue_slots',
                                          'dialogue_has_valid_financial_task': True}))
    return result


def sql_rows(raw, table):
    for path in sorted(raw.glob('load_' + table + '*.dump')):
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                line = line.strip()
                if line.startswith('INSERT INTO'):
                    line = line.split('VALUES', 1)[1].strip()
                if line.startswith('('):
                    value = line[1:].removesuffix(',').removesuffix(';').removesuffix(')')
                    yield next(csv.reader([value], quotechar="'", doublequote=True))


def load_employees(raw):
    employees = {}
    for row in sql_rows(raw, 'employees'):
        emp, birthday, first, last, gender, hire = row
        name = first + ' ' + last
        if natural_name(name):
            employees[emp] = {'employee_id': emp, 'birth_date': birthday,
                              'FIRST': first, 'MIDDLE': '', 'LAST': last,
                              'name': name, 'hire_date': hire, '_employee_row': row,
                              '_event_rows': {}}
    # Keep a deterministic generous candidate reservoir; do not materialize 300k expanded records.
    chosen = sorted(employees, key=stable)[:40000]
    employees = {key: employees[key] for key in chosen}
    departments = {r[0]: r[1] for r in sql_rows(raw, 'departments')}
    snapshot = '2000-01-01'
    for table, key, start_i, end_i in [('salaries', 'salary', 2, 3), ('titles', 'title', 2, 3), ('dept_emp', 'department', 2, 3)]:
        for row in sql_rows(raw, table):
            emp = employees.get(row[0])
            if emp is not None and row[start_i] <= snapshot < row[end_i]:
                emp[key] = departments.get(row[1], row[1]) if key == 'department' else row[1]
                emp['_event_rows'][key] = {'table': table, 'row': row,
                                           'joined_value': emp[key]}
    result = []
    for emp_id, emp in employees.items():
        if not all(emp.get(f) for f in ('salary', 'department', 'title')):
            continue
        values = {f: emp[f] for f in ('name', 'birth_date', 'salary', 'hire_date', 'department', 'title')}
        text = '\n'.join(f'{f}: {v}' for f, v in values.items())
        ref = 'Employees:' + emp_id
        fields = {f: annotation(text, v, ref, locator='the structured employee record',
                               source_row=emp['_event_rows'].get(f, {'table': 'employees', 'row': emp['_employee_row']}),
                               derived=f == 'name')
                  for f, v in values.items()}
        result.append(candidate('Employees', emp_id, 'hr', 'structured', text, fields,
                                {k: emp[k] for k in ('FIRST', 'MIDDLE', 'LAST')},
                                {'snapshot_date': snapshot, 'salary_unit': 'original test_db integer; source does not specify currency or period',
                                 'syntheticity': 'fictional SQL test database'}))
    return result


def load_aml(raw, count=80000):
    c = connection(raw)
    path = str(raw / 'HI-Small_Trans.csv')
    rows = c.execute('''select "From Bank","Account","To Bank","Account_1",
        "Amount Paid","Payment Currency","Amount Received","Receiving Currency","Timestamp","Payment Format"
        from read_csv(?,all_varchar=true)
        order by hash("Timestamp","Account","Account_1","Amount Paid",17) limit ?''', [path, count]).fetchall()
    result = []
    for a, an, b, bn, paid, pc, received, rc, when, method in rows:
        if (a, an) == (b, bn):
            continue
        values = {'sender_account': a + ' / ' + an, 'recipient_account': b + ' / ' + bn,
                  'amount_paid': paid + ' ' + pc, 'amount_received': received + ' ' + rc,
                  'timestamp': when, 'payment_format': method}
        source_row = {'From Bank': a, 'Account': an, 'To Bank': b, 'Account_1': bn,
                      'Amount Paid': paid, 'Payment Currency': pc, 'Amount Received': received,
                      'Receiving Currency': rc, 'Timestamp': when, 'Payment Format': method}
        uid = fingerprint(json.dumps(source_row, sort_keys=True))[:24]
        result.append({'source_ref': 'IBM-AML:' + uid, 'values': values, 'source_row': source_row})
    return result


def load_synthea(csv_dir, work):
    c = connection(work)
    patients = {}
    with (csv_dir / 'patients.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            name = ' '.join(row[k] for k in ('FIRST', 'MIDDLE', 'LAST') if row.get(k))
            if not natural_name(name):
                continue
            patients[row['Id']] = {'row': row, 'name': name, 'events': collections.defaultdict(list)}
    # Keep representative literal events per patient, preserving original rows and relationships.
    specs = [('conditions', 'diagnosis'), ('medications', 'medication'),
             ('procedures', 'procedure'), ('allergies', 'allergy')]
    for table, field in specs:
        path = csv_dir / (table + '.csv')
        query = 'select * from read_csv(?,all_varchar=true)'
        cur = c.execute(query, [str(path)])
        cols = [d[0] for d in cur.description]
        while batch := cur.fetchmany(10000):
            for values in batch:
                row = dict(zip(cols, values))
                p = patients.get(row['PATIENT'])
                if p is None or not row.get('DESCRIPTION'):
                    continue
                if field == 'diagnosis' and not row['DESCRIPTION'].endswith('(disorder)'):
                    continue
                vals = p['events'][field]
                if len(vals) < 30 and row['DESCRIPTION'] not in {x['DESCRIPTION'] for x in vals}:
                    vals.append(row)
    cur = c.execute('select * from read_csv(?,all_varchar=true)', [str(csv_dir / 'observations.csv')])
    cols = [d[0] for d in cur.description]
    while batch := cur.fetchmany(20000):
        for values in batch:
            row = dict(zip(cols, values))
            p = patients.get(row['PATIENT'])
            if p is None or not row.get('VALUE') or row.get('TYPE') != 'numeric':
                continue
            if len(p['events']['observation']) < 12:
                p['events']['observation'].append(row)
    # Only encounter IDs used by the retained events need to be kept.
    needed = {r['ENCOUNTER'] for p in patients.values() for vals in p['events'].values() for r in vals if r.get('ENCOUNTER')}
    encounters = {}
    cur = c.execute('select * from read_csv(?,all_varchar=true)', [str(csv_dir / 'encounters.csv')])
    cols = [d[0] for d in cur.description]
    while batch := cur.fetchmany(20000):
        for values in batch:
            row = dict(zip(cols, values))
            if row['Id'] in needed:
                encounters[row['Id']] = row
    result, profiles = [], []
    used = collections.Counter()
    ordered = sorted(patients.items(), key=lambda item: (not bool(item[1]['events'].get('allergy')), stable(item[0])))
    for pid, p in ordered:
        row = p['row']
        ref = 'Synthea:' + pid
        basic = {'name': p['name'], 'birth_date': row['BIRTHDATE'], 'address': row['ADDRESS'], 'identity_number': row['SSN']}
        profile = {'source_ref': ref, 'values': basic, 'source_name_fields': {k: row.get(k, '') for k in ('FIRST', 'MIDDLE', 'LAST')},
                   'source_row': row, 'age': 2026 - int(row['BIRTHDATE'][:4]), 'gender': row.get('GENDER'),
                   'alive_at_reference': not row.get('DEATHDATE')}
        profiles.append(profile)
        if not p['events'].get('diagnosis'):
            continue
        values = dict(basic)
        source_rows = {f: row for f in basic}
        for field, events in p['events'].items():
            event = min(events, key=lambda r: (used[(field, r.get('DESCRIPTION', ''))], stable(json.dumps(r, sort_keys=True))))
            if field == 'observation':
                value = event['DESCRIPTION'] + ': ' + event['VALUE'] + ((' ' + event['UNITS']) if event.get('UNITS') else '')
            else:
                value = event['DESCRIPTION']
            values[field] = value
            source_rows[field] = event
            used[(field, event.get('DESCRIPTION', ''))] += 1
            if field == 'diagnosis' and event.get('ENCOUNTER') in encounters:
                enc = encounters[event['ENCOUNTER']]
                values['encounter_start'] = enc['START']
                source_rows['encounter_start'] = enc
        text = 'Selected original events from one synthetic patient history.\n' + '\n'.join(f'{f}: {v}' for f, v in values.items())
        fields = {f: annotation(text, value, ref, locator='the selected synthetic patient history', source_row=source_rows[f],
                               source_evidence=(source_rows[f].get('DESCRIPTION') or value),
                               derived=f in {'name', 'observation'}) for f, value in values.items() if value}
        result.append(candidate('Synthea', pid, 'medical', 'structured', text, fields,
                                profile['source_name_fields'], {'clinical_events_same_patient': True,
                                                               'events_can_be_from_different_encounters': True}))
    return result, profiles
