"""Offline vocabulary assembly. No patient rows or prediction files are exported."""
import csv
import re
from .config import file_hash, fingerprint

RESOURCE_VERSION = 'synthea_nlm_components_v1'


def assemble_resources(source_path, nlm, provenance):
    conditions = []
    with source_path.open(encoding='utf-8-sig', newline='') as f:
        unique = {(r['DESCRIPTION'], r['SYSTEM'], r['CODE']) for r in csv.DictReader(f)
                  if r['DESCRIPTION'].endswith(' (disorder)')}
    for name, system, code in sorted(unique):
        conditions.append({'id': f'{system}:{code}', 'name': name, 'aliases': [],
                           'codes': [[system, code]], 'source': 'synthea'})
    # Known misspelled display strings are not promoted into real-disease negatives.
    bad = {'Streptococcal pharyngits', 'Periodonitis - acute',
           'Respiratoy syncytial virus (RSV) infection'}
    excluded = []
    for r in sorted(nlm, key=lambda r: str(r['key_id'])):
        name = r['primary_name']
        if r.get('is_procedure') or name in bad:
            excluded.append({'id': str(r['key_id']), 'name': name,
                             'reason': 'procedure' if r.get('is_procedure') else 'known_display_typo'})
            continue
        conditions.append({'id': 'NLM:' + str(r['key_id']), 'name': name,
                           'aliases': r.get('synonyms') or [], 'source': 'nlm',
                           'codes': [['ICD-10-CM', c.strip()] for c in (r.get('icd10cm_codes') or '').split(',') if c.strip()]})
    names = {c['name'] for c in conditions}
    bodies = {family: {} for family in ('sprain', 'fracture', 'infection', 'otitis', 'concussion')}
    def add(family, site, original):
        bodies[family].setdefault(site, []).append(original)
    for name in sorted(names):
        core = re.sub(r'\s+\(disorder\)$', '', name)
        m = re.fullmatch(r'Sprain of (.+)|(.+) sprain', core, re.I)
        if m:
            site = (m[1] or m[2]).casefold()
            add('sprain', {'cervical': 'neck', 'sacroiliac': 'sacroiliac joint'}.get(site, site), name)
        m = re.fullmatch(r'(?:Closed |Osteoporotic )?Fracture of (.+)|(.+) fracture', core, re.I)
        if m:
            site = re.split(r' with(?:out)? ', (m[1] or m[2]).casefold())[0]
            if not any(s in site for s in ('compression', 'stress', 'blowout', 'burst', '/')):
                site = {'jaw': 'mandible', 'vertebral': 'vertebral column', 'nasal': 'nasal bone',
                        'pelvic': 'pelvis', 'femur neck': 'femoral neck'}.get(site, site)
                add('fracture', site, name)
    fixed = {
        'infection': {'tooth': 'Infection of tooth (disorder)', 'leg': 'Leg infection',
                      'liver': 'Hepatic infection', 'urinary tract': 'Urinary tract infection (UTI)',
                      'upper respiratory tract': 'Upper respiratory infection'},
        'otitis': {'middle ear': 'Otitis media', 'external ear': 'Otitis externa'},
        'concussion': {'brain': 'Concussion injury of brain (disorder)'}}
    for family, values in fixed.items():
        for site, evidence in values.items():
            if evidence in names:
                add(family, site, evidence)
    body_groups = {family: [{'value': site, 'evidence': sorted(set(evidence))}
                            for site, evidence in sorted(values.items())] for family, values in bodies.items()}
    metadata = {'version': RESOURCE_VERSION, 'source_conditions_sha256': file_hash(source_path),
                'scope': 'all source diagnosis labels and external terminology; no test answers/predictions read',
                'external': provenance, 'excluded_external': excluded,
                'conditions': len(conditions), 'body_group_sizes': {k: len(v) for k, v in body_groups.items()},
                'limits': 'Terminology-derived components; heuristic compatibility groups, not clinician-reviewed individual counterfactuals.'}
    payload = {'metadata': metadata, 'conditions': conditions, 'body_groups': body_groups}
    return dict(payload, fingerprint=fingerprint(payload))
