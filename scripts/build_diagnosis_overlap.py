"""Revise only test diagnoses from the same patients' recorded source history."""
import argparse
import collections
import copy
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('name', 'birth_date', 'address', 'diagnosis')
SOURCE_KEYS = ('PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION', 'START', 'SYSTEM')


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_records(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def frozen_base(base):
    hashes = read_json(base / 'artifact_hashes.json')
    for name, expected in hashes.items():
        if sha(base / name) != expected:
            raise ValueError(f'Frozen base artifact changed: {name}')
    return hashes


def source_key(row):
    return tuple(row[k] for k in SOURCE_KEYS)


def record_key(record):
    p = record['provenance']
    return (p['source_patient_id'], p['source_encounter_id'], p['source_condition_code'],
            record['fields']['diagnosis'], p['source_condition_start'], p['source_condition_system'])


def read_source(source, records, expected_hashes):
    ids = {r['provenance']['source_patient_id'] for r in records}
    for table, expected in expected_hashes.items():
        if sha(source / f'{table}.csv') != expected:
            raise ValueError(f'Source snapshot changed: {table}')
    with (source / 'patients.csv').open(encoding='utf-8-sig', newline='') as f:
        patients = {p['Id']: p for p in csv.DictReader(f) if p['Id'] in ids}
    with (source / 'encounters.csv').open(encoding='utf-8-sig', newline='') as f:
        encounters = {e['Id']: e for e in csv.DictReader(f) if e['PATIENT'] in ids}
    conditions = {}
    with (source / 'conditions.csv').open(encoding='utf-8-sig', newline='') as f:
        for c in csv.DictReader(f):
            if c['PATIENT'] in ids and c['DESCRIPTION'].endswith('(disorder)'):
                e = encounters.get(c['ENCOUNTER'])
                if e is not None and e['PATIENT'] == c['PATIENT']:
                    conditions[source_key(c)] = c
    return patients, encounters, conditions


def choose_conditions(test, train, encounters, conditions, percent, seed):
    if type(percent) is not int or not 0 <= percent <= 100:
        raise ValueError('test_seen_diagnosis_percent must be an integer in [0, 100]')
    labels = {r['fields']['diagnosis'] for r in train}
    codes = {(r['provenance']['source_condition_system'], r['provenance']['source_condition_code']) for r in train}
    options = collections.defaultdict(list)
    for c in conditions.values():
        if c['DESCRIPTION'] not in labels and (c['SYSTEM'], c['CODE']) not in codes:
            options[c['PATIENT']].append(c)
    forced = {r['provenance']['source_patient_id'] for r in test if not options[r['provenance']['source_patient_id']]}
    seen_target = len(test) * percent // 100
    if len(forced) > seen_target:
        raise ValueError(f'Cannot satisfy cap without changing patients: {len(forced)} unavoidable seen / {len(test)}; allowed {seen_target}')
    def stable(pid):
        return hashlib.sha256(f'{seed}|keep-seen|{pid}'.encode()).hexdigest()
    optional = sorted((r['provenance']['source_patient_id'] for r in test
                       if r['provenance']['source_patient_id'] not in forced), key=stable)
    keep = forced | set(optional[:seen_target - len(forced)])
    selected = {}
    for r in test:
        pid = r['provenance']['source_patient_id']
        if pid in keep:
            selected[r['record_id']] = conditions[record_key(r)]
        else:
            selected[r['record_id']] = max(options[pid], key=lambda c: (
                encounters[c['ENCOUNTER']]['START'], c['ENCOUNTER'], c['CODE'], c['DESCRIPTION'], c['START']))
    return selected, len(forced)


def revised_record(original, condition, encounter, tokenizer):
    r = copy.deepcopy(original)
    old = original['fields']['diagnosis']
    label = condition['DESCRIPTION']
    if label == old:
        return r
    r['fields']['diagnosis'] = label
    marker = 'conditions.DESCRIPTION: '
    lines = r['private_prefix'].splitlines()
    assert sum(line.startswith(marker) for line in lines) == 1
    r['private_prefix'] = '\n'.join(marker + label if line.startswith(marker) else line for line in lines)
    next(qa for qa in r['qa'] if qa['field'] == 'diagnosis')['answer'] = label
    r['token_lengths']['private_prefix'] = len(tokenizer.encode(r['private_prefix'], add_special_tokens=False).ids)
    r['token_lengths']['answers']['diagnosis'] = len(tokenizer.encode(label, add_special_tokens=False).ids)
    r['provenance'].update(source_encounter_id=condition['ENCOUNTER'], source_condition_code=condition['CODE'],
                           source_condition_system=condition['SYSTEM'], source_condition_description=label,
                           source_condition_start=condition['START'], source_encounter_start=encounter['START'])
    return r


def audit(base, out, source, tokenizer, percent):
    """Validate invariant fields and joins independently of the selection rule."""
    original = read_records(base / 'test.jsonl')
    revised = read_records(out / 'test.jsonl')
    train = read_records(base / 'train.jsonl')
    labels = {r['fields']['diagnosis'] for r in train}
    codes = {(r['provenance']['source_condition_system'], r['provenance']['source_condition_code']) for r in train}
    manifest = read_json(base / 'manifest.json')
    patients, encounters, conditions = read_source(source, original, manifest['source_csv_sha256'])
    assert len(original) == len(revised) == 1000
    assert [r['record_id'] for r in original] == [r['record_id'] for r in revised]
    kept = []
    changed = []
    for before, after in zip(original, revised):
        pid = before['provenance']['source_patient_id']
        assert pid == after['provenance']['source_patient_id']
        assert tuple(after['fields']) == FIELDS
        assert all(before['fields'][f] == after['fields'][f] for f in FIELDS[:3])
        # Remove only explicitly permitted differences; everything else must match.
        left, right = copy.deepcopy(before), copy.deepcopy(after)
        for item in (left, right):
            item['fields'].pop('diagnosis')
            item['qa'] = [q for q in item['qa'] if q['field'] != 'diagnosis']
            item['private_prefix'] = '\n'.join(line for line in item['private_prefix'].splitlines()
                                              if not line.startswith('conditions.DESCRIPTION: '))
            item['token_lengths'].pop('private_prefix')
            item['token_lengths']['answers'].pop('diagnosis')
            for key in ('source_encounter_id', 'source_condition_code', 'source_condition_system',
                        'source_condition_description', 'source_condition_start', 'source_encounter_start'):
                item['provenance'].pop(key)
        assert left == right, 'A non-diagnosis field or metadata changed'
        assert next(q for q in before['qa'] if q['field'] == 'diagnosis')['question'] == next(q for q in after['qa'] if q['field'] == 'diagnosis')['question']
        p, c = patients[pid], conditions[record_key(after)]
        e = encounters[c['ENCOUNTER']]
        assert e['PATIENT'] == pid and p['BIRTHDATE'] <= e['START'][:10]
        expected_prefix = '\n'.join([f'patients.{k}: {p[k]}' for k in ('FIRST','MIDDLE','LAST')]
                                    + [f"patients.BIRTHDATE: {p['BIRTHDATE']}", f"patients.ADDRESS: {p['ADDRESS']}",
                                       f"conditions.DESCRIPTION: {c['DESCRIPTION']}"])
        assert after['private_prefix'] == expected_prefix
        assert all(q['answer'] == after['fields'][q['field']] for q in after['qa'])
        assert after['token_lengths']['private_prefix'] == len(tokenizer.encode(expected_prefix, add_special_tokens=False).ids)
        assert after['token_lengths']['answers'] == {f: len(tokenizer.encode(v, add_special_tokens=False).ids) for f,v in after['fields'].items()}
        seen = after['fields']['diagnosis'] in labels
        code_seen = (c['SYSTEM'],c['CODE']) in codes
        assert seen == code_seen
        if seen:
            assert before == after, 'Seen subset must preserve original diagnoses and metadata'
            kept.append(after['record_id'])
        else:
            assert before['fields']['diagnosis'] != after['fields']['diagnosis']
            assert after['provenance']['source_encounter_start'] == e['START']
            assert after['provenance']['source_condition_description'] == c['DESCRIPTION']
            changed.append(after['record_id'])
    assert len(kept) == len(revised) * percent // 100
    for split in ('dev', 'train', 'validation'):
        assert (base / f'{split}.jsonl').read_bytes() == (out / f'{split}.jsonl').read_bytes()
    assert (out / 'all_records.jsonl').read_bytes() == b''.join((out / f'{split}.jsonl').read_bytes() for split in ('dev','train','validation','test'))
    selected_sources = read_json(out / 'source_rows_selected.json')
    base_sources = read_json(base / 'source_rows_selected.json')
    for i, r in enumerate(read_records(out / 'all_records.jsonl')):
        p = r['provenance']
        assert selected_sources['patients'][i] == base_sources['patients'][i]
        if r['split'] == 'test':
            assert selected_sources['encounters'][i] == encounters[p['source_encounter_id']]
            assert selected_sources['conditions'][i] == conditions[record_key(r)]
        else:
            assert all(selected_sources[table][i] == base_sources[table][i] for table in ('patients','encounters','conditions'))
    distinct = {r['fields']['diagnosis'] for r in revised}
    return {'status': 'PASS', 'name_format': manifest['name_format'], 'test_records': len(revised), 'unchanged_patient_order_and_ids': True,
            'unchanged_name_birth_date_address_records': len(revised), 'source_joins_and_token_lengths': 'PASS',
            'unchanged_splits_byte_identical': ['dev','train','validation'],
            'diagnosis_train_seen_records': len(kept), 'diagnosis_train_unseen_records': len(changed),
            'diagnosis_overlap_percent_by_patient': len(kept) / len(revised) * 100,
            'code_overlap_percent_by_patient': len(kept) / len(revised) * 100,
            'diagnosis_distinct_values': len(distinct), 'shared_diagnosis_distinct_values': len(distinct & labels),
            'distinct_label_overlap_percent': len(distinct & labels) / len(distinct) * 100,
            'seen_record_ids': kept, 'unseen_record_ids': changed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/diagnosis_overlap10.json')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    cfg = read_json(resolve(args.config))
    base, out, source = (resolve(cfg[k]) for k in ('base_dataset_directory','output_directory','source_csv_directory'))
    frozen_base(base)
    original_manifest = read_json(base / 'manifest.json')
    from tokenizers import Tokenizer
    tokenizer_path = ROOT / 'tools/tokenizer/Qwen3-4B/tokenizer.json'
    assert sha(tokenizer_path) == original_manifest['tokenizer']['sha256']
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    percent = cfg['test_seen_diagnosis_percent']
    if args.validate_only:
        report = audit(base, out, source, tokenizer, percent)
        for name, expected in read_json(out / 'artifact_hashes.json').items():
            assert sha(out / name) == expected, name
        print(json.dumps({k:v for k,v in report.items() if not k.endswith('_record_ids')}, ensure_ascii=False, indent=2))
        return
    eval_config_path = resolve(cfg['evaluation_config'])
    if out.exists() or eval_config_path.exists():
        raise FileExistsError('Revision/config already exists; validate it or select fresh destinations')
    original = read_records(base / 'test.jsonl')
    train = read_records(base / 'train.jsonl')
    patients, encounters, conditions = read_source(source, original, original_manifest['source_csv_sha256'])
    selected, unavoidable = choose_conditions(original, train, encounters, conditions, percent, cfg['seed'])
    revised = [revised_record(r, selected[r['record_id']], encounters[selected[r['record_id']]['ENCOUNTER']], tokenizer) for r in original]
    out.mkdir(parents=True, exist_ok=False)
    for split in ('dev','train','validation'):
        shutil.copyfile(base / f'{split}.jsonl', out / f'{split}.jsonl')
    write_records(out / 'test.jsonl', revised)
    (out / 'all_records.jsonl').write_bytes(b''.join((out / f'{s}.jsonl').read_bytes() for s in ('dev','train','validation','test')))
    selected_sources = read_json(base / 'source_rows_selected.json')
    for i, r in enumerate(read_records(out / 'all_records.jsonl')):
        if r['split'] == 'test':
            condition = selected[r['record_id']]
            selected_sources['conditions'][i] = condition
            selected_sources['encounters'][i] = encounters[condition['ENCOUNTER']]
    dump(out / 'source_rows_selected.json', selected_sources)
    manifest = copy.deepcopy(original_manifest)
    manifest.update(dataset_id='synthea_medical_privacy_natural_names_diagnosis_overlap10_v1',
                    base_dataset_id=original_manifest['dataset_id'], record_dataset_id=original_manifest['dataset_id'],
                    revision_date='2026-10-17', diagnosis_label_overlap_required=False,
                    build_config_sha256=sha(resolve(args.config)), build_script_sha256=sha(__file__),
                    base_artifact_hashes_sha256=sha(base / 'artifact_hashes.json'),
                    revision_scope='Only test diagnoses and linked encounter/condition provenance; original patient identities and other three fields unchanged.',
                    diagnosis_selection={'test_seen_percent_by_patient':percent, 'seed':cfg['seed'],
                                         'unavoidable_seen_patients':unavoidable, 'selection':cfg['selection'],
                                         'reference':cfg['reference']},
                    splitting='All original split memberships and patient order retained.',
                    status='Built and independently checked against the frozen source rows.')
    dump(out / 'manifest.json', manifest)
    dump(out / 'diagnosis_labels.json', {split:sorted({r['fields']['diagnosis'] for r in read_records(out/f'{split}.jsonl')}) for split in ('dev','train','validation','test')})
    report = audit(base, out, source, tokenizer, percent)
    dump(out / 'validation_report.json', report)
    changes = [{'record_id':before['record_id'], 'source_patient_id':before['provenance']['source_patient_id'],
                'old_diagnosis':before['fields']['diagnosis'], 'new_diagnosis':after['fields']['diagnosis'],
                'old_source_encounter_id':before['provenance']['source_encounter_id'],
                'new_source_encounter_id':after['provenance']['source_encounter_id'],
                'new_condition_code':after['provenance']['source_condition_code']}
               for before, after in zip(original,revised) if before['fields']['diagnosis'] != after['fields']['diagnosis']]
    write_records(out / 'diagnosis_changes.jsonl', changes)
    card = f'''# 诊断训练取值重叠率10%的测试版本

原测试集1000位患者及顺序不变，仅900位患者的诊断改选为同一患者原始病史中训练未见的诊断。姓名、完整生日、地址1000/1000逐条不变；dev/train/validation文件逐字节不变。原dataset_v1及原结果保留。

重叠定义：测试患者的完整诊断DESCRIPTION在原训练集出现的人数/1000。已见100人（10%），未见900人（90%）；SYSTEM+CODE口径也为10%。病名类别去重后的交集比例是另一种统计：{report['shared_diagnosis_distinct_values']}/{report['diagnosis_distinct_values']}，即{report['distinct_label_overlap_percent']:.2f}%，不与患者口径混用。

{unavoidable}位患者没有训练未见的合格病史诊断，保留其原诊断；另{report['diagnosis_train_seen_records'] - unavoidable}位按固定种子患者ID哈希保留原诊断。其余患者选择最近一次有合格未见诊断的就诊，固定就诊ID、代码、描述及起始日期打破平局。只使用原以(disorder)结尾的DESCRIPTION；不翻译或制造新病名，不把诊断称为主诊断或当前活动诊断。选择过程不读取predictions.csv或模型分数。

private_prefix只更新conditions.DESCRIPTION行，QA只更新诊断答案，问题措辞不变。溯源同步更新到真实关联的condition和encounter行，重新计算诊断及前缀token长度。记录中的dataset_id保留原始记录来源，manifest的dataset_id标识本次评估版本。

训练未见仅指本次共享引导优化的监督取值未见，不代表基础模型预训练未见。该版本是看到原测试结果后新增的补充评估，不替代或覆盖原测试结果。诊断改变会改变编码缓存，因此需重新生成全部问题的回答，不能直接沿用原CSV中的姓名、生日和地址预测。

使用configs/evaluation_diagnosis_overlap10.json加载本目录，并评估自然姓名版重新训练的best.pt；旧编号姓名检查点不兼容。禁止将补充测试反馈用于宣称原独立测试结果的调参。构建配置位于configs/diagnosis_overlap10.json，构建/只读复核入口为scripts/build_diagnosis_overlap.py及其--validate-only参数。
'''
    (out / '数据集说明.md').write_text(card, encoding='utf-8')
    dump(out / 'artifact_hashes.json', {p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    # Validate through the real loader before exposing the evaluation config.
    import sys
    sys.path.insert(0, str(ROOT))
    from medical_kv.data import dataset_identity, load_split
    evaluation = read_json(resolve(cfg['reference_experiment_config']))
    evaluation.update(dataset_directory=cfg['output_directory'], output_directory='runs/test_birth_parts_diag_overlap10')
    dataset_identity(evaluation)
    assert len(load_split(evaluation,'test')) == 1000
    dump(eval_config_path,evaluation)
    frozen_base(base)
    print(json.dumps({'status':'PASS', 'dataset_directory':str(out), 'evaluation_config':str(eval_config_path),
                      **{k:v for k,v in report.items() if not k.endswith('_record_ids')}},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
