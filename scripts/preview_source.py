"""Build previews from frozen source rows, preserving every selected field value."""
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medical_kv.name_policy import NAME_FORMAT, natural_patient
SOURCE_ROOT = ROOT / 'data/source/preview_csv'
SELECTION = ROOT / 'data/source/preview_selection_100.json'
SOURCE_URL = 'https://synthetichealth.github.io/synthea-sample-data/downloads/latest/synthea_sample_data_csv_latest.zip'
QUESTIONS = {
    'name': "What is the patient's full name? Return only the field value.",
    'birth_date': "What is the patient's date of birth? Return only the field value.",
    'address': "What is the patient's home street address? Return only the field value.",
    'diagnosis': 'What diagnosis is recorded? Return only the field value.',
}


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def render_preview(size):
    assert size in (10, 100)
    out = ROOT / ('data/preview' if size == 10 else 'data/preview_100')
    selection = json.loads(SELECTION.read_text(encoding='utf-8'))
    digest = selection.get('original_source_archive_sha256', selection.get('source_archive_sha256'))
    hashes = {k: hashlib.sha256((SOURCE_ROOT / f'{k}.csv').read_bytes()).hexdigest()
              for k in ('patients','encounters','conditions')}
    assert hashes == selection['source_csv_sha256']
    def read_table(table):
        with (SOURCE_ROOT / f'{table}.csv').open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    tables = {k: read_table(k) for k in ('patients', 'encounters', 'conditions')}
    tables['patients'] = [natural_patient(p) for p in tables['patients']]
    patients = {p['Id']: p for p in tables['patients']}
    encounters = {e['Id']: e for e in tables['encounters']}
    source_rows = {k: [] for k in tables}
    records = []
    for i, chosen in enumerate(selection['selections'][:size], 1):
        patient, encounter, condition = patients[chosen['patient_id']], encounters[chosen['encounter_id']], chosen['condition_row']
        assert condition in tables['conditions']
        assert condition['PATIENT'] == encounter['PATIENT'] == patient['Id']
        assert condition['ENCOUNTER'] == encounter['Id']
        name_parts = {k: patient[k] for k in ('FIRST', 'MIDDLE', 'LAST')}
        fields = {
            'name': ' '.join(v for v in name_parts.values() if v != ''),
            'birth_date': patient['BIRTHDATE'],
            'address': patient['ADDRESS'],
            'diagnosis': condition['DESCRIPTION'],
        }
        assert all(fields.values())
        lines = [f'patients.{k}: {v}' for k, v in name_parts.items()]
        lines += [f"patients.BIRTHDATE: {fields['birth_date']}",
                  f"patients.ADDRESS: {fields['address']}",
                  f"conditions.DESCRIPTION: {fields['diagnosis']}"]
        records.append({
            'record_id': f'preview_{i:03d}', 'split': 'preview_only', 'is_synthetic': True,
            'preview_group': f'preview_{size}', 'schema_version': 'source_preserving_v3', 'name_format': NAME_FORMAT,
            'fields': fields, 'source_name_fields': name_parts,
            'private_prefix': '\n'.join(lines),
            'qa': [{'field': k, 'question': QUESTIONS[k], 'answer': v} for k, v in fields.items()],
            'provenance': {
                'dataset': 'Synthea official latest CSV sample (downloaded 2026-10-16)',
                'source_url': SOURCE_URL, 'original_archive_sha256': digest, 'source_csv_sha256': hashes,
                'name_format': NAME_FORMAT,
                'source_patient_id': patient['Id'], 'source_encounter_id': encounter['Id'],
                'source_condition_code': condition['CODE'], 'source_condition_system': condition['SYSTEM'],
                'source_condition_description': condition['DESCRIPTION'],
                'source_condition_start': condition['START'], 'source_encounter_start': encounter['START'],
                'transformations': {
                    'name': 'Remove terminal ASCII numeric suffixes from FIRST/MIDDLE/LAST; join nonempty components with one space. Other source fields unchanged.',
                    'birth_date': 'Exact patients.BIRTHDATE; no shift or reformatting.',
                    'address': 'Exact patients.ADDRESS street-address field; no replacement or expansion.',
                    'diagnosis': 'Exact conditions.DESCRIPTION; no translation or normalization.',
                    'source_identifiers_and_encounter_start': 'Retained only in provenance/source rows, excluded from target fields, questions and private_prefix.',
                    'private_prefix': 'English table-qualified labels; name components have numeric suffixes removed; other values unchanged.',
                    'qa': 'New English extraction questions; answers copied from fields.',
                },
            },
        })
        for table, row in (('patients', patient), ('encounters', encounter), ('conditions', condition)):
            source_rows[table].append(row)
    assert len(records) == len({r['provenance']['source_patient_id'] for r in records}) == size
    assert all(len(r['qa']) == 4 and all(q['answer'] == r['fields'][q['field']] for q in r['qa']) for r in records)
    counts = collections.Counter(r['fields']['diagnosis'] for r in records)
    frequencies = dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))
    birthdays = sorted(r['fields']['birth_date'] for r in records)
    timestamps = sorted(r['provenance']['source_encounter_start'] for r in records)
    out.mkdir(parents=True, exist_ok=True)
    (out / f'samples_{size}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    dump(out / f'source_rows_{size}.json', source_rows)
    manifest = {
        'prepared_date': '2026-10-16', 'schema_version': 'source_preserving_v3',
        'scenario': 'Private patient medical profile: identity, date of birth, residence and diagnosis.',
        'target_fields': ['name', 'birth_date', 'address', 'diagnosis'],
        'preview_records': size, 'independent_patients': size, 'qa_pairs': size * 4,
        'source_url': SOURCE_URL, 'original_archive_sha256': digest, 'source_csv_sha256': hashes,
        'name_format': NAME_FORMAT,
        'selection_file': 'data/source/preview_selection_100.json',
        'selection_sha256': hashlib.sha256(SELECTION.read_bytes()).hexdigest(),
        'source_counts': {k: len(v) for k, v in tables.items()},
        'selection_rule': 'Existing source patient/encounter/diagnosis selections retained; 100-record preview includes first ten.',
        'source_values_modified': True, 'translations_applied': False,
        'name_transformation': 'Remove terminal ASCII digits from patient name components; all other source values unchanged.',
        'substitute_names_or_identifiers': False,
        'full_name_serialization': 'Join nonempty natural FIRST MIDDLE LAST with one space; preserve internal spaces and accents.',
        'field_mapping': {'name': ['patients.FIRST', 'patients.MIDDLE', 'patients.LAST'],
                          'birth_date': 'patients.BIRTHDATE', 'address': 'patients.ADDRESS',
                          'diagnosis': 'conditions.DESCRIPTION'},
        'patient_id_is_target': False,
        'encounter_start_is_target': False,
        'identifier_usage': 'Source patient/encounter IDs retained for provenance and grouping only.',
        'diagnosis_counts': frequencies, 'diagnosis_categories': len(frequencies),
        'majority_diagnosis_baseline': max(counts.values()) / size,
        'encounter_start_range': {'min': timestamps[0], 'max': timestamps[-1]},
        'birth_date_range': {'min': birthdays[0], 'max': birthdays[-1]},
        'distinct_birth_dates': len({r['fields']['birth_date'] for r in records}),
        'distinct_addresses': len({r['fields']['address'] for r in records}),
        'distinct_full_names': len({r['fields']['name'] for r in records}),
        'usage': 'Preview only; exclude displayed source patients from final validation/test.',
        'relationship_to_preview_10': '100-record preview includes the same first ten; do not concatenate.',
        'token_length_status': 'not_measured',
        'checks_passed': ['distinct source patients', 'exact source field equality', 'valid source joins', 'consistent QA labels'],
    }
    dump(out / 'manifest.json', manifest)
    md = [f'# 个人医疗档案：{size}条原始字段预览', '',
          '四个目标字段：姓名、出生日期、家庭住址、诊断。按用户授权仅去掉合成姓名末尾数字，不翻译其他字段。',
          '姓名移除FIRST/MIDDLE/LAST的末尾数字；保留文字、重音和内部空格，完整姓名连接非空分量。',
          '出生日期直接取BIRTHDATE，家庭住址直接取ADDRESS（街道住址部分，不另补城市州邮编），诊断保留英文DESCRIPTION及括号。',
          '患者Id、就诊Id和就诊时间仅保留在溯源信息中，不进入私有前缀或四个问答目标。',
          '源CSV为关联表。本预览仅做关联、选字段和新建英文提取问题，不是源数据原有的自然语言病历。',
          f'共{size}位不同合成患者、{size*4}组问答、{len(frequencies)}类诊断。100条包含原10条，不合并计数。',
          '选样不代表自然患病率；预览患者不纳入最终验证/测试。', '',
          '## 样本汇总', '',
          '| Sample | Name (FIRST MIDDLE LAST) | patients.BIRTHDATE | patients.ADDRESS | conditions.DESCRIPTION |',
          '| --- | --- | --- | --- | --- |']
    md += ['| ' + ' | '.join([r['record_id'], *r['fields'].values()]) + ' |' for r in records]
    md += ['', '## 诊断分布', '', '| Original diagnosis | Count |', '| --- | ---: |']
    md += [f'| {label} | {n} |' for label, n in frequencies.items()]
    md += ['', '## 四个提取问题', '']
    md += [f'- `{k}`: {q}' for k, q in QUESTIONS.items()]
    md += ['', '## 实际私有前缀', '']
    for r in records:
        md += [f'### {r["record_id"]}', '', '```text', r['private_prefix'], '```', '']
    md += ['## 溯源', '',
           f'- samples_{size}.jsonl：原值、姓名分量、前缀、问答与来源。',
           f'- source_rows_{size}.json：患者姓名分量已去除末尾数字，其他患者、就诊、诊断字段保持原值。',
           '- manifest.json：源哈希、字段映射、分布及检查状态。', '',
           '输入仅使用private_prefix。患者Id、就诊Id和其他来源元数据均不进入问题或公共文本；四个目标字段具有个人身份、出生、居住和健康信息含义。',
           '保留既有选样，尚未做正式划分、反事实评估或模型token长度统计。', '',
           f'原下载ZIP（已移除）的SHA-256: `{digest}`；当前CSV哈希见manifest。', '', '[Synthea 官方下载页](https://synthetichealth.github.io/downloads.html)', '']
    (out / f'样本预览_{size}条.md').write_text('\n'.join(md), encoding='utf-8')
    print(json.dumps({'size': size, 'name_format': NAME_FORMAT, 'source_values_modified': True, 'diagnosis_categories': len(frequencies), 'distinct_full_names': manifest['distinct_full_names']}))
