import csv
import json
import pytest
from medical_kv.config import ROOT, load_config, file_hash, fingerprint
from medical_kv.data import dataset_identity, load_split
from medical_kv.io import inference_identity, checkpoint_compatibility
from medical_kv.name_policy import NAME_FORMAT, NAME_KEYS, natural_component, natural_patient, has_name_digits
from test_experiment import setup


def test_name_transform_preserves_accents_spaces_and_nonname_numbers():
    assert natural_component('José Emilio366') == 'José Emilio'
    assert natural_component('Ronnie7') == 'Ronnie'
    assert natural_component('') == ''
    patient = {'FIRST':'Ronnie7','MIDDLE':'','LAST':'Koch169','MAIDEN':'Smith23',
               'BIRTHDATE':'2001-01-01','ADDRESS':'759 Kirlin Lock Unit 3','Id':'patient123'}
    natural = natural_patient(patient)
    assert natural['FIRST'] == 'Ronnie' and natural['LAST'] == 'Koch' and natural['MAIDEN'] == 'Smith'
    assert all(natural[k] == patient[k] for k in ('BIRTHDATE','ADDRESS','Id'))
    assert natural_patient(natural) == natural


def test_every_active_split_and_preview_has_no_numbered_names():
    for config in ('configs/experiment_v2_birth_parts.json','configs/evaluation_diagnosis_overlap10.json'):
        cfg=load_config(config)
        identity=dataset_identity(cfg)
        assert cfg['name_format'] == NAME_FORMAT
        assert 'natural_names' in identity['dataset_id']
        for split in ('dev','train','validation','test'):
            rows=load_split(cfg,split)
            for row in rows:
                assert row['name_format'] == NAME_FORMAT
                assert not has_name_digits(row['fields']['name'])
                assert all(not has_name_digits(v) for v in row['source_name_fields'].values())
                assert row['fields']['name'] == ' '.join(row['source_name_fields'][k] for k in NAME_KEYS if row['source_name_fields'][k])
                assert all(q['answer'] == row['fields'][q['field']] for q in row['qa'])
    for folder,size in (('preview',10),('preview_100',100)):
        rows=[json.loads(line) for line in (ROOT/'data'/folder/f'samples_{size}.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows) == size
        assert all(not has_name_digits(row['fields']['name']) for row in rows)


def test_source_names_clean_and_obsolete_dataset_packages_removed():
    for folder in ('data/source/candidates_5000/csv','data/source/preview_csv'):
        with (ROOT/folder/'patients.csv').open(encoding='utf-8-sig',newline='') as stream:
            patients=list(csv.DictReader(stream))
        assert patients
        assert all(not has_name_digits(p[key]) for p in patients for key in (*NAME_KEYS,'MAIDEN') if key in p)
    report=json.loads((ROOT/'data/name_naturalization_report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'PASS' and report['retraining_required'] is True
    for filename in report['obsolete_packages_removed']:
        assert not (ROOT/filename).exists()


def test_name_format_enters_checkpoint_protocol_and_mixed_data_is_rejected(setup):
    backend,cfg,_=setup
    current=inference_identity(backend)
    cfg['name_format']='legacy_numbered_names'
    assert inference_identity(backend) != current
    mixed=load_config('configs/experiment_v2_birth_parts.json')
    mixed['name_format']='legacy_numbered_names'
    with pytest.raises(ValueError,match='name format mismatch'):
        dataset_identity(mixed)


def historical_checkpoint(backend):
    # Reproduce the pre-migration hash independently of the compatibility helper.
    cfg = dict(backend.cfg)
    cfg.pop('name_format')
    signature = fingerprint({'model': backend.identity, 'protocol': {k: cfg[k] for k in (
        'prompt_length', 'public_text', 'separator', 'question_format', 'enable_thinking',
        'dtype', 'attention_implementation')}, 'public_ids': backend.public_ids,
        'question_template': backend.tokenizer.chat_template})
    return {'inference_signature': signature, 'run_info': {'config': cfg}}


def test_legacy_name_transfer_requires_explicit_opt_in_and_marks_results(setup):
    backend, cfg, _ = setup
    checkpoint = historical_checkpoint(backend)
    with pytest.raises(ValueError, match='allow-name-format-transfer'):
        checkpoint_compatibility(checkpoint, backend)
    result = checkpoint_compatibility(checkpoint, backend, allow_name_format_transfer=True)
    assert result['name_format_transfer'] is True
    assert result['checkpoint_name_format'] == 'legacy_numbered_names'
    assert result['evaluation_name_format'] == NAME_FORMAT
    assert 'no retraining' in result['evaluation_note']
    current = {'inference_signature': inference_identity(backend), 'run_info': {'config': dict(cfg)}}
    assert checkpoint_compatibility(current, backend)['name_format_transfer'] is False


@pytest.mark.parametrize('changed', ['model', 'tokenizer', 'public_ids', 'prompt_length',
    'public_text', 'separator', 'question_format', 'enable_thinking', 'dtype',
    'attention_implementation', 'birth_date_mode', 'unknown_name_format'])
def test_transfer_does_not_bypass_other_protocol_checks(setup, changed):
    backend, cfg, _ = setup
    checkpoint = historical_checkpoint(backend)
    if changed == 'model':
        backend.identity = {'model': 'different-model'}
    elif changed == 'tokenizer':
        backend.tokenizer.chat_template = 'different-template'
    elif changed == 'public_ids':
        backend.public_ids = backend.public_ids + [17]
    elif changed == 'birth_date_mode':
        cfg['birth_date_mode'] = 'parts'
    elif changed == 'unknown_name_format':
        checkpoint['run_info']['config']['name_format'] = 'unknown_format'
    else:
        cfg[changed] = 'changed'
    with pytest.raises(ValueError, match='protocol mismatch'):
        checkpoint_compatibility(checkpoint, backend, allow_name_format_transfer=True)


def test_transfer_option_cannot_be_used_to_resume_training(capsys):
    from medical_kv.cli import _main
    with pytest.raises(SystemExit):
        _main(['train-soft', '--resume', 'old.pt', '--allow-name-format-transfer'])
    assert 'cannot resume training' in capsys.readouterr().err
