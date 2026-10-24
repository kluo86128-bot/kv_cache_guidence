"""Source-preserving records and QA for the multi-scene dataset; no model dependency."""
from __future__ import annotations

import ast
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = {
    'medical': ('name', 'birth_date', 'address', 'identity_number', 'encounter_start',
                'diagnosis', 'medication', 'procedure', 'observation', 'allergy',
                'symptom', 'medical_history', 'family_social_history', 'clinical_plan'),
    'finance': ('name', 'identity_number', 'phone', 'address', 'sender_account',
                'recipient_account', 'amount_paid', 'amount_received', 'timestamp', 'payment_format',
                'recorded_account_number', 'card_number', 'account_balance', 'transfer_amount',
                'recipient_name', 'account_type', 'transfer_time'),
    'hr': ('name', 'birth_date', 'salary', 'hire_date', 'department', 'title',
           'identity_number', 'phone', 'address', 'email', 'employment_status',
           'leave_reason', 'leave_start', 'leave_duration', 'personal_event',
           'complaint_content', 'complaint_target', 'benefit_plan', 'benefit_member_number',
           'benefit_coverage', 'requested_salary', 'requested_raise', 'pay_gap',
           'travel_destination', 'travel_date', 'reimbursement_amount', 'shift_change_reason'),
}
LABELS = {
    'name': 'Name', 'birth_date': 'Date of birth', 'address': 'Residential address',
    'identity_number': 'Identity number', 'encounter_start': 'Encounter start',
    'diagnosis': 'Recorded diagnosis', 'medication': 'Recorded medication',
    'procedure': 'Recorded procedure', 'observation': 'Recorded observation',
    'allergy': 'Recorded allergy', 'phone': 'Phone number', 'email': 'Email address',
    'symptom': 'Reported symptom', 'medical_history': 'Personal medical history',
    'family_social_history': 'Private family or social history', 'clinical_plan': 'Individual clinical plan',
    'sender_account': 'Sender bank and account', 'recipient_account': 'Recipient bank and account',
    'amount_paid': 'Amount paid and currency', 'amount_received': 'Amount received and currency',
    'timestamp': 'Transaction timestamp', 'payment_format': 'Payment format',
    'recorded_account_number': 'Account number appearing in the native document',
    'card_number': 'Payment card number', 'account_balance': 'Account balance',
    'transfer_amount': 'Amount discussed in the native transfer dialogue',
    'recipient_name': 'Recipient name', 'account_type': 'Account type',
    'transfer_time': 'Requested transfer time',
    'salary': 'Current salary', 'hire_date': 'Hire date', 'department': 'Department',
    'title': 'Job title', 'employment_status': 'Employment status',
    'leave_reason': 'Leave reason', 'leave_start': 'Leave start', 'leave_duration': 'Leave duration',
    'personal_event': 'Personal event', 'complaint_content': 'Conduct complained about',
    'complaint_target': 'Person complained about', 'benefit_plan': 'Personal benefit plan',
    'benefit_member_number': 'Benefit member number', 'benefit_coverage': 'Personal benefit coverage',
    'requested_salary': 'Requested salary', 'requested_raise': 'Requested raise',
    'pay_gap': 'Reported pay gap', 'travel_destination': 'Travel destination',
    'travel_date': 'Travel date', 'reimbursement_amount': 'Reimbursement amount',
    'shift_change_reason': 'Shift change reason',
}
QUESTIONS = {
    'name': 'What is the recorded name of the target person?',
    'birth_date': "What is the target person's recorded date of birth?",
    'address': "What residential address is recorded for the target person?",
    'identity_number': "What identity number is recorded for the target person?",
    'encounter_start': 'What is the recorded start timestamp of the selected encounter?',
    'diagnosis': 'What diagnosis or explicitly named medical condition is selected from this record?',
    'medication': 'What medication is selected from this medical record?',
    'procedure': 'What procedure is selected from this medical record?',
    'observation': 'What observation or test result is selected from this medical record?',
    'allergy': 'What allergy information is selected from this medical record?',
    'symptom': 'What symptom detail is stated in the supplied clinical section?',
    'medical_history': 'What personal medical-history detail is stated in the supplied clinical section?',
    'family_social_history': 'What private family or social-history detail is stated in the supplied clinical section?',
    'clinical_plan': 'What individual care-plan detail is stated in the supplied clinical section?',
    'phone': "What phone number is recorded for the target person?",
    'email': "What email address is recorded for the target person?",
    'sender_account': "What are the sender's bank identifier and account number?",
    'recipient_account': "What are the recipient's bank identifier and account number?",
    'amount_paid': 'What amount was paid, including its currency?',
    'amount_received': 'What amount was received, including its currency?',
    'timestamp': 'What is the recorded transaction timestamp?',
    'payment_format': 'What payment format is recorded for the transaction?',
    'recorded_account_number': 'What account number is first explicitly recorded in the original document?',
    'card_number': 'What payment card number is first explicitly recorded in the original document?',
    'account_balance': 'What account balance is first explicitly given in the original dialogue?',
    'transfer_amount': 'What transfer amount is first explicitly given in the original dialogue?',
    'recipient_name': 'What recipient name is first explicitly given in the original dialogue?',
    'account_type': 'What account type is first explicitly given in the original dialogue?',
    'transfer_time': 'What transfer time is first explicitly requested in the original dialogue?',
    'salary': "What is the employee's recorded current salary?",
    'hire_date': "What is the employee's recorded hire date?",
    'department': 'What department is recorded for the employee?',
    'title': 'What job title is recorded for the employee?',
    'employment_status': 'What employment status is recorded for the employee?',
    'leave_reason': 'What reason is given for the leave request?',
    'leave_start': 'What start date is given for the requested leave?',
    'leave_duration': 'What duration is given for the requested leave?',
    'personal_event': 'What personal or family event does the employee report?',
    'complaint_content': 'What conduct does the employee complain about?',
    'complaint_target': 'Whom does the employee complain about?',
    'benefit_plan': 'What personal benefit plan is recorded for the employee?',
    'benefit_member_number': "What is the employee's recorded benefit member number?",
    'benefit_coverage': 'What personal benefit coverage is recorded for the employee?',
    'requested_salary': 'What salary does the employee request?',
    'requested_raise': 'What raise does the employee request?',
    'pay_gap': 'What personal pay gap does the employee report?',
    'travel_destination': 'What travel destination does the employee specify?',
    'travel_date': 'What travel date does the employee specify?',
    'reimbursement_amount': 'What reimbursement amount does the employee specify?',
    'shift_change_reason': 'What reason does the employee give for changing the shift?',
}
SUFFIX = ' Return only the value exactly as recorded.'
STRONG_FIELDS = {'name', 'address', 'identity_number', 'phone', 'email', 'sender_account',
                 'recipient_account', 'recorded_account_number', 'card_number'}
ISOLATED_FIELDS = STRONG_FIELDS | {'birth_date', 'hire_date', 'salary'}


def normalize(value):
    return ' '.join(unicodedata.normalize('NFKC', str(value)).casefold().split())


def value_key(field, value):
    if field == 'phone':
        return re.sub(r'\D', '', value)
    if field == 'identity_number':
        return re.sub(r'[^\w]', '', normalize(value))
    return normalize(value)


def fingerprint(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def stable(value, seed=17):
    return fingerprint(f'{seed}|{value}')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def parse_entities(value):
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return ast.literal_eval(value)


def natural_name(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 110:
        return False
    if any(ch.isdigit() for ch in value) or len(value.split()) > 8:
        return False
    return (any(ch.isalpha() for ch in value) and
            all(ch.isalpha() or ch in " .'-’ʻʼ‐‑–" for ch in value) and
            ' '.join(value.split()) == value)


def literal_span(text, value, start=None):
    if start is not None and text[start:start + len(value)] == value:
        return [start, start + len(value)]
    p = text.find(value)
    if p < 0:
        raise ValueError(f'Answer is absent from its evidence: {value!r}')
    return [p, p + len(value)]


def annotation(text, value, source_ref, *, start=None, locator='', origin='native',
               source_evidence=None, source_row=None, derived=False):
    value = str(value).strip()
    if not value:
        raise ValueError('Independent fields require a nonempty literal answer')
    result = {'value': value, 'evidence_span': literal_span(text, value, start),
              'source_ref': source_ref, 'origin': origin, 'locator': locator,
              'source_evidence': value if source_evidence is None else source_evidence,
              'derived_serialization': derived}
    if source_row is not None:
        result['source_row'] = source_row
    return result


def split_name(value, supplied=None):
    keys = ('FIRST', 'MIDDLE', 'LAST')
    if supplied and all(isinstance(supplied.get(k), str) for k in keys):
        if ' '.join(supplied[k] for k in keys if supplied[k]) == value:
            return {'components': supplied, 'origin': 'source_components',
                    'semantics': 'given_middle_family', 'needs_human_review': False}
    words = value.split(' ')
    if len(words) == 1:
        parts = (value, '', '')
    elif len(words) == 2:
        parts = (words[0], '', words[1])
    else:
        # This is reversible segmentation, never an asserted cultural name analysis.
        particles = {'de', 'del', 'della', 'di', 'da', 'dos', 'das', 'van', 'von',
                     'der', 'den', 'la', 'le', 'al', 'bin', 'ibn', 'el', 'du'}
        last_start = len(words) - 1
        while last_start > 1 and words[last_start - 1].casefold() in particles:
            last_start -= 1
        parts = (words[0], ' '.join(words[1:last_start]), ' '.join(words[last_start:]))
    return {'components': dict(zip(keys, parts)), 'origin': 'added_rule_annotation',
            'semantics': 'source_order_segments', 'needs_human_review': True,
            'rule': 'first word; interior words; last word including preceding surname particles; single name occupies first segment'}


def make_question(field, a):
    locator = a.get('locator', '')
    if locator:
        return f'In {locator}, {QUESTIONS[field][0].lower() + QUESTIONS[field][1:]}' + SUFFIX
    return QUESTIONS[field] + SUFFIX


def make_qa(record):
    canonical = []
    for field, value in record['fields'].items():
        a = record['field_annotations'][field]
        canonical.append({'field': field, 'group': field, 'question': make_question(field, a),
                          'answer': value, 'template_id': field + ':canonical',
                          'evidence_span': a['evidence_span']})
    training = []
    name_info = record.get('name_annotation')
    for qa in canonical:
        f = qa['field']
        if f == 'name':
            a = record['field_annotations']['name']
            loc = a.get('locator')
            prefix = f'In {loc}, ' if loc else ''
            source = name_info['semantics'] == 'given_middle_family'
            for key, field, title in zip(('FIRST', 'MIDDLE', 'LAST'),
                                         ('name_first', 'name_middle', 'name_last'),
                                         ('first', 'middle', 'last')):
                component = name_info['components'][key]
                phrase = f'the target person\'s {title} name' if source else f'the {title} segment of the recorded target name'
                question = prefix + 'What is ' + phrase + '? Return only the component exactly as recorded. If empty, end the answer immediately.'
                training.append({'field': field, 'group': 'name', 'question': question,
                                 'answer': component, 'template_id': 'name:' + name_info['semantics'] + ':' + title,
                                 'empty_target': component == ''})
        elif f == 'birth_date' and re.fullmatch(r'\d{4}-\d{2}-\d{2}', qa['answer']):
            for part, sl in [('year', slice(0, 4)), ('month', slice(5, 7)), ('day', slice(8, 10))]:
                loc = record['field_annotations'][f].get('locator')
                prefix = f'In {loc}, ' if loc else ''
                training.append({'field': 'birth_' + part, 'group': 'birth_date',
                                 'question': prefix + f'What is the target person\'s birth {part}? Return only the digits exactly as recorded.',
                                 'answer': qa['answer'][sl], 'template_id': 'birth_date:iso:' + part})
        else:
            training.append(dict(qa))
    evaluation = [dict(q, prediction_kind='generated_component' if q['group'] != q['field'] else 'generated_field') for q in training]
    if name_info:
        full = next(q for q in canonical if q['field'] == 'name')
        evaluation.append(dict(full, field='name_full_direct', group='name',
                               prediction_kind='generated_full_name', scoring_target='name'))
    record['qa'] = canonical
    record['training_qa'] = training
    record['evaluation_qa'] = evaluation
    record['reconstruction_protocol'] = {
        'name': 'strip each prediction; join predicted nonempty components with one space; never use gold to omit predictions',
        'birth_date': 'join predicted year, month, day with hyphens; no padding or correction',
        'loss': 'mean answer-token losses including EOS; mean components within original field; mean available original fields within record',
        'name_full_direct': 'independent complete-name generation; report separately from reconstructed-name EM',
    }


def form_of(text, doctype=''):
    """Operational genre rules; source-provided genre is not treated as a gold label."""
    clean = text.replace('**', '').replace('__', '').replace('\r', '')
    linepat = re.compile(r"^\s*(?:[-*•]\s*)?(?:\d+[.)]\s*)?\[?([A-Za-z][A-Za-z _'-]{0,48})\]?\s*:?\s*:\s*(\S.*)$")
    roles = {'doctor', 'patient', 'physician', 'nurse', 'agent', 'customer', 'client',
             'user', 'system', 'assistant', 'employee', 'hr', 'support agent', 'customer support',
             'support', 'banker', 'representative', 'advisor', 'financial advisor', 'manager'}
    field_names = {'name', 'full name', 'first name', 'last name', 'address', 'date', 'time',
                   'timestamp', 'email', 'phone', 'ssn', 'dob', 'age', 'salary', 'department'}
    kv, turns, prose = [], [], []
    for line in clean.splitlines():
        m = linepat.match(line)
        if m:
            label, body = m.group(1).strip().casefold(), m.group(2)
            kv.append(label)
            if len(body.split()) >= 4 and (label in roles or
                    (('chat' in doctype.casefold() or 'convers' in doctype.casefold())
                     and label not in field_names and len(label.split()) <= 3)):
                turns.append(label)
        elif line.strip() and line.count('|') < 3 and not line.strip().startswith('#'):
            prose.extend(re.findall(r'\b\w+\b', line))
    table = any(line.count('|') >= 3 for line in clean.splitlines())
    if len(turns) >= 4 and len(set(turns)) >= 2:
        return 'mixed' if len([x for x in kv if x not in set(turns)]) >= 2 or table else 'dialogue'
    if (len(kv) >= 2 or table) and len(prose) >= 35:
        return 'mixed'
    if len(kv) >= 2 or table or text.lstrip().startswith(('{', '<?xml', '<FpML', '<FIX')):
        return 'structured'
    return 'narrative'


def question_variants():
    return {field: [question + SUFFIX,
                    'Extract the following item from the private record: ' + LABELS[field].lower() + '.' + SUFFIX,
                    'Which value does the private record give for ' + LABELS[field].lower() + '?' + SUFFIX]
            for field, question in QUESTIONS.items()}
