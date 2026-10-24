"""Data integrity tests for the diagnosis-only supplemental test revision."""
import json
from pathlib import Path
import pytest
from medical_kv.config import ROOT, load_config, file_hash
from medical_kv.data import dataset_identity, load_split
from medical_kv.io import inference_identity
from test_experiment import setup


@pytest.fixture(scope='module')
def revision():
    path = ROOT / 'configs/evaluation_diagnosis_overlap10.json'
    if not path.exists():
        pytest.skip('Supplemental diagnosis dataset has not been built on this host')
    cfg = load_config(path)
    base_cfg = load_config('configs/experiment_v2_birth_parts.json')
    return cfg, base_cfg, load_split(base_cfg, 'test'), load_split(cfg, 'test'), load_split(base_cfg, 'train')


def test_only_diagnosis_revised_and_training_files_unchanged(revision):
    cfg, base_cfg, original, revised, train = revision
    dataset_identity(cfg)
    dataset_identity(base_cfg)
    assert len(original) == len(revised) == 1000
    for split in ('dev','train','validation'):
        assert file_hash(ROOT / cfg['dataset_directory'] / f'{split}.jsonl') == file_hash(ROOT / base_cfg['dataset_directory'] / f'{split}.jsonl')
    for before, after in zip(original, revised):
        assert before['record_id'] == after['record_id']
        assert before['provenance']['source_patient_id'] == after['provenance']['source_patient_id']
        for field in ('name','birth_date','address'):
            assert before['fields'][field] == after['fields'][field]
            assert next(q for q in before['qa'] if q['field']==field) == next(q for q in after['qa'] if q['field']==field)
        assert [line for line in before['private_prefix'].splitlines() if not line.startswith('conditions.DESCRIPTION: ')] == [line for line in after['private_prefix'].splitlines() if not line.startswith('conditions.DESCRIPTION: ')]
    assert not {r['provenance']['source_patient_id'] for r in revised} & {r['provenance']['source_patient_id'] for r in train}


def test_ten_percent_by_patients_and_source_rows_follow_new_diagnosis(revision):
    cfg, _, original, revised, train = revision
    labels = {r['fields']['diagnosis'] for r in train}
    codes = {(r['provenance']['source_condition_system'],r['provenance']['source_condition_code']) for r in train}
    assert sum(r['fields']['diagnosis'] in labels for r in revised) == 100
    assert sum((r['provenance']['source_condition_system'],r['provenance']['source_condition_code']) in codes for r in revised) == 100
    assert sum(a['fields']['diagnosis'] != b['fields']['diagnosis'] for a,b in zip(original,revised)) == 900
    assert all(a == b for a,b in zip(original,revised) if b['fields']['diagnosis'] in labels)
    # Birthdays were explicitly outside this change: the original 44/1000 remains.
    train_dates = {r['fields']['birth_date'] for r in train}
    assert sum(r['fields']['birth_date'] in train_dates for r in revised) == 44
    folder = ROOT / cfg['dataset_directory']
    sources = json.loads((folder / 'source_rows_selected.json').read_text(encoding='utf-8'))
    all_records = [json.loads(line) for line in (folder / 'all_records.jsonl').read_text(encoding='utf-8').splitlines()]
    for i,r in enumerate(all_records):
        p,c,e = sources['patients'][i],sources['conditions'][i],sources['encounters'][i]
        assert c['PATIENT'] == e['PATIENT'] == p['Id'] == r['provenance']['source_patient_id']
        assert c['ENCOUNTER'] == e['Id'] == r['provenance']['source_encounter_id']
        assert c['DESCRIPTION'] == r['fields']['diagnosis'] == r['provenance']['source_condition_description']


def test_evaluation_revision_preserves_checkpoint_input_protocol(setup, revision):
    backend, _, _ = setup
    cfg, base_cfg, _, _, _ = revision
    # New test records and dataset hashes do not change the model/tokenizer/input
    # protocol: the existing learned U remains a valid evaluation checkpoint.
    backend.cfg = base_cfg
    original_signature = inference_identity(backend)
    backend.cfg = cfg
    assert inference_identity(backend) == original_signature
    changed_keys = {key for key in base_cfg if base_cfg[key] != cfg[key]}
    assert changed_keys == {'dataset_directory','output_directory'}
