"""In-place, user-authorized removal of Synthea numeric name suffixes.

Never retain numbered-name dataset backups. Historical experiment outputs are
not rewritten; their labels and scores belong to the obsolete input format.
"""
import collections
import copy
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from medical_kv.name_policy import NAME_FORMAT, NAME_KEYS, has_name_digits, natural_component, natural_patient

SPLITS = ('dev', 'train', 'validation', 'test')
TRANSFORMATION = ('Remove terminal ASCII digits from FIRST/MIDDLE/LAST/MAIDEN; '
                  'join nonempty FIRST/MIDDLE/LAST with one space. Preserve letters, accents, '
                  'internal spaces, dates, addresses and diagnoses; no translation.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def write_records(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def refresh_hashes(folder):
    dump(folder / 'artifact_hashes.json', {p.name:sha(p) for p in sorted(folder.iterdir())
                                          if p.is_file() and p.name != 'artifact_hashes.json'})


def write_csv(path, columns, rows):
    temporary = path.with_name(path.name + '.natural.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def naturalize_patient_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns, original = reader.fieldnames, list(reader)
    updated = [natural_patient(row) for row in original]
    for a,b in zip(original, updated):
        assert {k:v for k,v in a.items() if k not in (*NAME_KEYS,'MAIDEN')} == {k:v for k,v in b.items() if k not in (*NAME_KEYS,'MAIDEN')}
        assert all(not has_name_digits(b[k]) for k in (*NAME_KEYS,'MAIDEN') if k in b)
    write_csv(path,columns,updated)
    return updated, sum(a[k] != b[k] for a,b in zip(original,updated) for k in (*NAME_KEYS,'MAIDEN') if k in a)


def describe(values):
    values=sorted(values)
    return {'min':values[0], 'mean':round(statistics.mean(values),3), 'median':statistics.median(values),
            'p95':values[math.ceil(.95*len(values))-1], 'max':values[-1]}


def naturalize_record(original, dataset_id, source_hashes, tokenizer):
    r=copy.deepcopy(original)
    r['source_name_fields']={k:natural_component(v) for k,v in r['source_name_fields'].items()}
    r['fields']['name']=' '.join(r['source_name_fields'][k] for k in NAME_KEYS if r['source_name_fields'][k])
    lines=[]
    for line in r['private_prefix'].splitlines():
        key=next((k for k in NAME_KEYS if line.startswith(f'patients.{k}: ')),None)
        lines.append(f'patients.{key}: {r["source_name_fields"][key]}' if key else line)
    r['private_prefix']='\n'.join(lines)
    for qa in r['qa']:
        if qa['field']=='name': qa['answer']=r['fields']['name']
    r['dataset_id']=dataset_id
    r['name_format']=NAME_FORMAT
    r['provenance']['source_csv_sha256']=source_hashes
    r['provenance']['name_format']=NAME_FORMAT
    r['provenance']['transformations']=TRANSFORMATION
    r['token_lengths']={'private_prefix':len(tokenizer.encode(r['private_prefix'],add_special_tokens=False).ids),
                       'answers':{k:len(tokenizer.encode(v,add_special_tokens=False).ids) for k,v in r['fields'].items()}}
    assert all(r['fields'][f] == original['fields'][f] for f in ('birth_date','address','diagnosis'))
    assert r['record_id']==original['record_id'] and r['split']==original['split']
    assert r['provenance']['source_patient_id']==original['provenance']['source_patient_id']
    assert all(not has_name_digits(v) for v in r['source_name_fields'].values())
    assert not has_name_digits(r['fields']['name'])
    for field in ('birth_date','address','diagnosis'):
        assert next(q for q in r['qa'] if q['field']==field)==next(q for q in original['qa'] if q['field']==field)
    assert '\n'.join(line for line in original['private_prefix'].splitlines() if not any(line.startswith(f'patients.{key}: ') for key in NAME_KEYS)) == '\n'.join(line for line in r['private_prefix'].splitlines() if not any(line.startswith(f'patients.{key}: ') for key in NAME_KEYS))
    return r


def check_collisions(split_rows, preview_rows):
    import unicodedata
    def norm(s): return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
    sets={split:{norm(r['fields']['name']) for r in rows} for split,rows in split_rows.items()}
    preview_names={norm(r['fields']['name']) for r in preview_rows}
    for split,rows in split_rows.items():
        assert len(sets[split])==len(rows),f'Within-split name collision: {split}'
        assert not sets[split]&preview_names, f'Preview name collision: {split}'
    overlaps=[]
    for i,a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            count=len(sets[a]&sets[b])
            assert count==0, f'Cross-split name collision: {a}/{b}'
            overlaps.append({'splits':[a,b],'normalized_names':count})
    return overlaps


def main():
    from tokenizers import Tokenizer
    report_path=ROOT/'data/name_naturalization_report.json'
    if report_path.exists() and load(report_path).get('status')=='PASS':
        raise FileExistsError('Naturalization already completed. Run validation scripts instead of rewriting it.')
    folders=[ROOT/'data/dataset_v1',ROOT/'data/dataset_diagnosis_overlap10']
    original_data={folder.name:{split:records(folder/f'{split}.jsonl') for split in SPLITS} for folder in folders}
    original_manifests={folder.name:load(folder/'manifest.json') for folder in folders}
    for folder in folders:
        for filename,expected in load(folder/'artifact_hashes.json').items():
            assert sha(folder/filename)==expected, f'Existing artifact mismatch: {folder.name}/{filename}'
    source=ROOT/'data/source/candidates_5000/csv'
    old_source_hashes={table:sha(source/f'{table}.csv') for table in ('patients','encounters','conditions')}
    assert old_source_hashes==original_manifests['dataset_v1']['source_csv_sha256']
    # Check new name collisions before the first mutation.
    preview_original=records(ROOT/'data/preview_100/samples_100.jsonl')
    def planned(r):
        r=copy.deepcopy(r)
        r['fields']['name']=' '.join(natural_component(r['source_name_fields'][key]) for key in NAME_KEYS if r['source_name_fields'][key])
        return r
    check_collisions({s:[planned(r) for r in rs] for s,rs in original_data['dataset_v1'].items()},[planned(r) for r in preview_original])
    patients, changed_source_fields=naturalize_patient_csv(source/'patients.csv')
    source_hashes={table:sha(source/f'{table}.csv') for table in ('patients','encounters','conditions')}
    assert all(old_source_hashes[t]==source_hashes[t] for t in ('encounters','conditions'))
    # Replace the numbered official preview archive with three plain CSV tables.
    archive=ROOT/'data/source/synthea_sample_data_csv_latest.zip'
    preview_source=ROOT/'data/source/preview_csv'
    preview_source.mkdir(exist_ok=True)
    selection_path=ROOT/'data/source/preview_selection_100.json'
    selection=load(selection_path)
    original_archive_hash=sha(archive)
    assert original_archive_hash==selection['source_archive_sha256']
    with zipfile.ZipFile(archive) as zipped:
        for table in ('patients','encounters','conditions'):
            member=next(n for n in zipped.namelist() if n.rsplit('/',1)[-1]==f'{table}.csv')
            (preview_source/f'{table}.csv').write_bytes(zipped.read(member))
    _, changed_preview_fields=naturalize_patient_csv(preview_source/'patients.csv')
    selection['original_source_archive_sha256']=selection.pop('source_archive_sha256')
    selection['source_csv_sha256']={table:sha(preview_source/f'{table}.csv') for table in ('patients','encounters','conditions')}
    selection['name_format']=NAME_FORMAT
    dump(selection_path,selection)
    import preview_source as preview_module
    preview_module.render_preview(10)
    preview_module.render_preview(100)
    preview_rows=records(ROOT/'data/preview_100/samples_100.jsonl')
    tokenizer=Tokenizer.from_file(str(ROOT/'tools/tokenizer/Qwen3-4B/tokenizer.json'))
    dataset_id='synthea_medical_privacy_natural_names_v1'
    transformed={}
    for folder in folders:
        split_rows={s:[naturalize_record(r,dataset_id,source_hashes,tokenizer) for r in rs]
                    for s,rs in original_data[folder.name].items()}
        transformed[folder.name]=split_rows
        collisions=check_collisions(split_rows,preview_rows)
        for split,rows in split_rows.items(): write_records(folder/f'{split}.jsonl',rows)
        write_records(folder/'all_records.jsonl',[r for s in SPLITS for r in split_rows[s]])
        sources=load(folder/'source_rows_selected.json')
        sources['patients']=[natural_patient(p) for p in sources['patients']]
        dump(folder/'source_rows_selected.json',sources)
        m=original_manifests[folder.name]
        m['dataset_id']=dataset_id if folder.name=='dataset_v1' else 'synthea_medical_privacy_natural_names_diagnosis_overlap10_v1'
        if folder.name!='dataset_v1':
            m.update(base_dataset_id=dataset_id,record_dataset_id=dataset_id)
        m.update(name_format=NAME_FORMAT, name_transformation=TRANSFORMATION, source_values_modified=True,
                 source_csv_sha256=source_hashes, name_revision_date='2026-10-17',
                 name_revision_report='data/name_naturalization_report.json')
        m['grouping']='Original patient membership preserved; normalized natural full names and addresses rechecked for cross-split collisions.'
        dump(folder/'manifest.json',m)
        audit=load(folder/'validation_report.json')
        audit.update(name_format=NAME_FORMAT, names_with_numeric_suffix=0,
                     name_transformation=TRANSFORMATION, name_cross_split_intersections=collisions)
        dump(folder/'validation_report.json',audit)
        stats={}
        for split,rows in split_rows.items():
            stats[split]={'records':len(rows),'qa_pairs':len(rows)*4,
                         'diagnosis_counts':dict(sorted(collections.Counter(r['fields']['diagnosis'] for r in rows).items())),
                         'birth_decade_counts':dict(sorted(collections.Counter(r['fields']['birth_date'][:3]+'0s' for r in rows).items())),
                         'private_prefix_tokens':describe([r['token_lengths']['private_prefix'] for r in rows]),
                         'answer_tokens':{field:describe([r['token_lengths']['answers'][field] for r in rows]) for field in ('name','birth_date','address','diagnosis')},
                         'distinct_values':{field:len({r['fields'][field] for r in rows}) for field in ('name','birth_date','address','diagnosis')}}
        dump(folder/'statistics.json',stats)
        if folder.name!='dataset_v1':
            card=folder/'数据集说明.md'
            card.write_text(card.read_text(encoding='utf-8')+'\n## 姓名自然化更新\n\n所有集合的FIRST/MIDDLE/LAST末尾数字已按用户要求移除，患者、生日、地址及本版本诊断选择不变。训练、验证、开发文件与当前自然姓名基础版逐字节一致。旧编号姓名检查点不能用于本版，需重新训练后评估。\n',encoding='utf-8')
        refresh_hashes(folder)
    # Keep all future generation and CLI configurations on the same name policy.
    cfg_path=ROOT/'configs/dataset_v1.json'
    cfg=load(cfg_path)
    cfg.update(dataset_id=dataset_id,name_format=NAME_FORMAT,name_transformation=TRANSFORMATION)
    dump(cfg_path,cfg)
    outputs={'experiment_v1.json':'runs/soft_natural_names_seed17',
             'experiment_v2_birth_parts.json':'runs/soft_natural_names_birth_parts_seed17_100epochs',
             'evaluation_diagnosis_overlap10.json':'runs/test_natural_names_diag_overlap10'}
    for filename,output in outputs.items():
        path=ROOT/'configs'/filename
        cfg=load(path)
        cfg.update(name_format=NAME_FORMAT,output_directory=output)
        dump(path,cfg)
    for filename in ('configs/synthea_candidates_5000.properties','data/source/synthea_embedded_defaults.properties'):
        path=ROOT/filename
        path.write_text(path.read_text(encoding='utf-8').replace('generate.append_numbers_to_person_names = true',
                                                              'generate.append_numbers_to_person_names = false'),encoding='utf-8')
    base=folders[0]
    # Regenerate examples and descriptive documentation; never create a ZIP.
    import finalize_dataset
    finalize_dataset.main()
    validation=subprocess.run([sys.executable,str(ROOT/'scripts/validate_dataset.py'),'--write-report'],
                              cwd=ROOT,text=True,capture_output=True,encoding='utf-8')
    if validation.returncode:
        raise RuntimeError(validation.stdout+'\n'+validation.stderr)
    assert json.loads(validation.stdout)['status']=='PASS'
    variant=folders[1]
    m=load(variant/'manifest.json')
    m['base_artifact_hashes_sha256']=sha(base/'artifact_hashes.json')
    dump(variant/'manifest.json',m)
    import build_diagnosis_overlap
    variant_audit=build_diagnosis_overlap.audit(base,variant,source,tokenizer,10)
    variant_audit.update(name_format=NAME_FORMAT,names_with_numeric_suffix=0,name_transformation=TRANSFORMATION)
    dump(variant/'validation_report.json',variant_audit)
    refresh_hashes(variant)
    # Delete only exact, obsolete dataset packages within this experiment.
    obsolete=('medical_privacy_prompt.zip','medical_privacy_prompt_v2.zip',
              'medical_privacy_prompt_v2.zip.sha256.json','实验代码与数据_v1.zip.sha256.json',
              'data/dataset_v1_2328.zip','data/dataset_v1_2328.zip.sha256.json',
              'data/source/synthea_sample_data_csv_latest.zip')
    removed=[]
    for filename in obsolete:
        path=ROOT/filename
        assert path.resolve().is_relative_to(ROOT.resolve())
        if path.exists():
            path.unlink()
            removed.append(filename)
    # Preserve experimental evidence, explicitly marked as the obsolete format.
    legacy={'status':'obsolete_for_current_dataset','reason':'Numeric name suffixes removed from all current datasets.',
            'name_format_of_existing_outputs':'legacy_numbered_names','current_name_format':NAME_FORMAT,
            'action':'Keep historical scores unchanged. Retrain in a fresh run directory; do not resume/evaluate old numbered-name checkpoints.'}
    dump(ROOT/'runs/LEGACY_NAME_FORMAT.json',legacy)
    report={'status':'PASS','name_format':NAME_FORMAT,'transformation':TRANSFORMATION,
            'authorization':'User requested in-place removal; no numbered-name dataset backups retained.',
            'source_patient_rows':len(patients),'source_name_fields_changed':changed_source_fields,
            'preview_name_fields_changed':changed_preview_fields,
            'original_source_csv_sha256':old_source_hashes,'current_source_csv_sha256':source_hashes,
            'original_preview_archive_sha256':original_archive_hash,
            'datasets':{folder.name:{'split_sizes':{s:len(rows) for s,rows in transformed[folder.name].items()},
                                   'name_digits_remaining':0,'other_three_fields_changed':0,
                                   'patient_membership_changed':0,'cross_split_name_collisions':0} for folder in folders},
            'previews':{'preview':10,'preview_100':100},
            'diagnosis_overlap10_seen_test_patients':variant_audit['diagnosis_train_seen_records'],
            'obsolete_packages_removed':removed,'historical_results_rewritten':False,
            'old_checkpoints_compatible':False,'retraining_required':True}
    dump(report_path,report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
