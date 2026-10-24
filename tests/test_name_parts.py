import copy
import csv
import json
from contextlib import contextmanager

import pytest
import torch

from test_experiment import setup
from medical_kv.config import ROOT, FIELDS, load_config, file_hash
from medical_kv.data import dataset_identity, load_split
from medical_kv.evaluation import evaluate_records, summarize, paired_compare
from medical_kv.io import read_checkpoint, inference_identity, checkpoint_compatibility
from medical_kv.tasks import NAME_PARTS, NAME_KEYS, BIRTH_PARTS, questions_for, grouped_loss


def name_record(record, middle=''):
    record = copy.deepcopy(record)
    source = dict(FIRST='Ax', MIDDLE=middle, LAST='Bo')
    record['source_name_fields'] = source
    record['fields']['name'] = ' '.join(source[k] for k in NAME_KEYS if source[k])
    for q in record['qa']:
        q['answer'] = record['fields'][q['field']]
    record['private_prefix'] = record['private_prefix'].replace('Name: Ax',
        '\n'.join(f'patients.{key}: {value}' for key, value in source.items()))
    return record


@pytest.mark.parametrize('config', ['configs/experiment_v4_name_birth_parts_no_negatives.json',
                                  'configs/evaluation_name_birth_parts_overlap10.json'])
def test_name_views_keep_all_frozen_data_and_empty_middle_values(config):
    cfg=load_config(config)
    before=dataset_identity(cfg)
    assert cfg['name_mode']==cfg['birth_date_mode']=='parts'
    assert cfg['diagnosis_contrastive']=={'enabled':False}
    for split in ('dev','train','validation','test'):
        rows=load_split(cfg,split)
        original=copy.deepcopy(rows)
        for r in rows:
            q={qa['field']:qa['answer'] for qa in questions_for(r,cfg)}
            assert len(q)==8
            assert [q[f] for f in NAME_PARTS]==[r['source_name_fields'][k] for k in NAME_KEYS]
            assert ' '.join(q[f] for f in NAME_PARTS if q[f])==r['fields']['name']
            assert '-'.join(q[f] for f in BIRTH_PARTS)==r['fields']['birth_date']
        assert rows==original
    assert dataset_identity(cfg)==before
    baseline=load_config('configs/experiment_v3_no_negatives.json')
    if 'experiment_' in config:
        assert {k for k in cfg if cfg[k]!=baseline[k]}=={'name_mode','diagnosis_contrastive','output_directory'}


@pytest.mark.parametrize('name_mode,birth_mode,count,rows_count', [
    ('full','full',4,4),('parts','full',6,7),('full','parts',6,7),('parts','parts',8,10)])
def test_all_name_and_birthday_protocol_combinations(setup, tmp_path, monkeypatch,
                                                     name_mode,birth_mode,count,rows_count):
    backend,cfg,r=setup
    cfg.update(name_mode=name_mode,birth_date_mode=birth_mode)
    r=name_record(r,middle='Mo')
    questions=questions_for(r,cfg)
    assert len(questions)==count
    monkeypatch.setattr(backend,'generate_many',lambda ctx,qs:[
        (q['answer'],{'generated_tokens':1,'stop_reason':'eos','truncated':False}) for q in qs])
    summary,rows=evaluate_records(backend,[r],prompt_ids=backend.initial_ids(),output=tmp_path/'evaluation')
    assert summary['qa_pairs']==count and summary['prediction_rows']==len(rows)==rows_count
    assert summary['name_mode']==name_mode and summary['birth_date_mode']==birth_mode
    assert summary['field_em']=={f:1. for f in FIELDS}
    assert summary['macro_em']==summary['all_fields_em']==summary['name_diagnosis_joint_em']==1.


def test_name_group_weight_and_gradient_including_empty_middle(setup):
    _,cfg,r=setup
    cfg.update(name_mode='parts',birth_date_mode='parts')
    questions=questions_for(name_record(r),cfg)
    losses=torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.],requires_grad=True)
    loss=grouped_loss(losses,questions)
    assert loss.item()==pytest.approx(((1+2+3)/3+(4+5+6)/3+7+8)/4)
    loss.backward()
    torch.testing.assert_close(losses.grad,torch.tensor([1/12]*6+[1/4]*2))
    with pytest.raises(ValueError,match='complete unique'):
        grouped_loss(losses[:-1],questions[:-1])
    with pytest.raises(ValueError,match='complete unique'):
        grouped_loss(losses,questions[:-1]+[questions[0]])


def test_eight_query_loss_and_prompt_gradient_match_serial_with_eos_only_middle(setup):
    backend,cfg,r=setup
    cfg.update(name_mode='parts',birth_date_mode='parts')
    r=name_record(r)
    questions=questions_for(r,cfg)
    assert questions[1]['field']=='name_middle' and questions[1]['answer']==''
    assert backend.question_and_target(questions[1])[1]==[backend.eos]
    with pytest.raises(ValueError,match='Empty target'):
        backend.question_and_target(dict(questions[-1],answer=''))
    prompt=backend.embedding(backend.tensor(backend.initial_ids()))[0].detach().clone().requires_grad_(True)
    context=backend.context(r,vectors=prompt)
    cached=[(k.detach().clone(),v.detach().clone()) for k,v in context.layers]
    sequential=torch.stack([backend.answer_loss(context,q) for q in questions])
    batched=backend.answer_losses(context,questions)
    torch.testing.assert_close(batched,sequential,rtol=1e-5,atol=2e-6)
    reference=torch.autograd.grad(grouped_loss(sequential,questions),prompt,retain_graph=True)[0]
    actual=backend.record_loss(r,vectors=prompt)
    gradient=torch.autograd.grad(actual,prompt)[0]
    torch.testing.assert_close(gradient,reference,rtol=3e-4,atol=3e-6)
    assert torch.isfinite(gradient).all() and gradient.norm()>0
    assert all(p.grad is None for p in backend.model.parameters())
    assert backend.generate_many(context,questions)==[backend.generate(context,q) for q in questions]
    for (k,v),(original_k,original_v) in zip(context.layers,cached):
        torch.testing.assert_close(k,original_k)
        torch.testing.assert_close(v,original_v)


@pytest.mark.parametrize('middle,generation,correct', [('', '', True),('', 'Wrong',False),('Mo','',False)])
def test_name_reconstruction_does_not_use_gold_to_omit_predicted_middle(
        setup,tmp_path,monkeypatch,middle,generation,correct):
    backend,cfg,r=setup
    cfg.update(name_mode='parts',birth_date_mode='parts')
    r=name_record(r,middle)
    def generate(context,qs):
        return [(generation if q['field']=='name_middle' else q['answer'],
                 {'generated_tokens':0 if q['field']=='name_middle' and not generation else 1,
                  'stop_reason':'eos','truncated':False}) for q in qs]
    monkeypatch.setattr(backend,'generate_many',generate)
    summary,rows=evaluate_records(backend,[r],prompt_ids=backend.initial_ids(),output=tmp_path/'evaluation')
    assert summary['qa_pairs']==8 and len(rows)==10
    assert summary['field_em']['name']==float(correct)
    assert summary['name_component_em']['name_middle']==float(correct)
    assert summary['macro_em']==(1. if correct else .75)
    name=next(x for x in rows if x['field']=='name')
    assert name['prediction']==' '.join(x for x in ['Ax',generation,'Bo'] if x)
    assert name['prediction_kind']=='reconstructed'
    assert name['nll']==pytest.approx(sum(x['nll'] for x in rows if x['field'] in NAME_PARTS)/3)
    assert summary['middle_name_subgroups']['empty' if middle=='' else 'nonempty']['records']==1
    with (tmp_path/'evaluation/predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        csv_rows=list(csv.DictReader(f))
    assert len(csv_rows)==10 and all(x['name_mode']=='parts' for x in csv_rows)
    assert next(x for x in csv_rows if x['field']=='name_middle')['prediction']==generation
    broken=copy.deepcopy(rows)
    next(x for x in broken if x['field']=='name')['prediction']='Invented'
    with pytest.raises(ValueError,match='reconstructed name'):
        summarize(broken)
    with pytest.raises(ValueError,match='Incomplete patient'):
        summarize([x for x in rows if x['field']!='name_middle'])


def test_name_sources_preserve_multiword_components_and_reject_inconsistent_labels(setup):
    _,cfg,r=setup
    cfg['name_mode']='parts'
    r=name_record(r)
    r['source_name_fields'].update(FIRST='María Elena',MIDDLE='',LAST='De La Cruz')
    r['fields']['name']='María Elena De La Cruz'
    qs=questions_for(r,cfg)
    assert [q['answer'] for q in qs[:3]]==['María Elena','','De La Cruz']
    r['fields']['name']='Other name'
    with pytest.raises(ValueError,match='source components'):
        questions_for(r,cfg)
    del r['source_name_fields']['MIDDLE']
    with pytest.raises(ValueError,match='original FIRST/MIDDLE/LAST'):
        questions_for(r,cfg)


def test_name_parts_configuration_does_not_change_public_cache_identity(setup):
    backend,cfg,_=setup
    checkpoint={'inference_signature':inference_identity(backend),'run_info':{'config':copy.deepcopy(cfg)}}
    cfg['name_mode']='parts'
    result=checkpoint_compatibility(checkpoint,backend)
    assert result['checkpoint_name_mode']=='full' and result['evaluation_name_mode']=='parts'
    assert result['name_protocol_transfer'] is True
    assert inference_identity(backend)==checkpoint['inference_signature']
    with pytest.raises(ValueError,match='name_mode'):
        load_config('configs/experiment_v4_name_birth_parts_no_negatives.json',{'name_mode':'invalid'})


@pytest.mark.parametrize('trained_mode,args,expected', [
    ('parts',[],'parts'),(None,[],'full'),('parts',['--name-mode','full'],'full'),
    ('parts',['--config','configs/evaluation_name_birth_parts_overlap10.json'],'parts')])
def test_evaluate_inherits_name_protocol_or_honors_explicit_choice(monkeypatch,trained_mode,args,expected):
    from types import SimpleNamespace
    from medical_kv import cli
    trained={'birth_date_mode':'parts'}
    if trained_mode is not None:trained['name_mode']=trained_mode
    monkeypatch.setattr(cli,'read_checkpoint',lambda p:{'run_info':{'config':trained}})
    monkeypatch.setattr(cli,'dataset_identity',lambda cfg:{'tokenizer_sha256':'fixture'})
    monkeypatch.setattr(cli,'initialize',lambda **kw:SimpleNamespace(active=False))
    monkeypatch.setattr(cli,'seed_all',lambda *a:None)
    class ReachedBackend(Exception):pass
    def load(cfg,*a):
        assert cfg['name_mode']==expected and cfg['birth_date_mode']=='parts'
        raise ReachedBackend
    monkeypatch.setattr(cli.Backend,'load',load)
    with pytest.raises(ReachedBackend):
        cli._main(['evaluate','--checkpoint','fixture.pt','--condition','soft',*args])


def test_name_parts_global_batch_gradient_with_empty_rank(setup):
    from medical_kv.training import local_batch_backward
    backend,cfg,r=setup
    cfg.update(name_mode='parts',birth_date_mode='parts')
    records=[dict(name_record(r,middle),record_id=str(i)) for i,middle in enumerate(['','Mo','Jo'])]
    initial=backend.embedding(backend.tensor(backend.initial_ids()))[0].detach().clone()
    p=initial.clone().requires_grad_(True)
    reference=sum(backend.record_loss(r,vectors=p) for r in records)/len(records)
    reference.backward()
    actual=torch.zeros_like(p)
    for rank in range(4):
        q=initial.clone().requires_grad_(True)
        local_batch_backward(backend,q,records,[0,1,2],rank,4)
        actual+=q.grad
    torch.testing.assert_close(actual,p.grad,rtol=3e-4,atol=3e-6)


def test_paired_comparison_marks_different_name_question_protocols(setup,tmp_path,monkeypatch):
    backend,cfg,r=setup
    cfg.update(name_mode='parts',birth_date_mode='parts')
    r=name_record(r)
    monkeypatch.setattr(backend,'generate_many',lambda ctx,qs:[
        (q['answer'],{'generated_tokens':1,'stop_reason':'eos','truncated':False}) for q in qs])
    evaluate_records(backend,[r],prompt_ids=backend.initial_ids(),output=tmp_path/'parts')
    cfg['name_mode']='full'
    evaluate_records(backend,[r],prompt_ids=backend.initial_ids(),output=tmp_path/'full')
    compared=paired_compare(tmp_path/'parts/predictions.jsonl',tmp_path/'full/predictions.jsonl',bootstrap_samples=10)
    assert compared['name_protocols_differ'] is True
    assert compared['name_protocol_a']=='parts' and compared['name_protocol_b']=='full'
    assert compared['macro_em_difference']['difference']==0.


def test_name_parts_training_never_builds_or_scores_negatives_and_resumes(setup,tmp_path,monkeypatch):
    import medical_kv.training as training
    import medical_kv.diagnosis_contrastive as negatives
    backend,cfg,r=setup
    cfg.update(name_mode='parts',birth_date_mode='parts',epochs=2,comparison_epochs=1,
               early_stopping=False,diagnosis_contrastive={'enabled':False})
    r=name_record(r)
    def forbidden(*a,**kw):raise AssertionError('Disabled negatives must not run')
    monkeypatch.setattr(negatives,'build_bank',forbidden)
    monkeypatch.setattr(negatives,'select_negatives',forbidden)
    monkeypatch.setattr(backend,'record_contrastive_loss',forbidden)
    class NoPlot:
        def __init__(self,*a,**kw):pass
        def update_training(self,*a):pass
        def update_validation(self,*a):pass
        def save(self):pass
    monkeypatch.setattr(training,'LossCurve',NoPlot)
    flag={'requested':False}
    original_evaluate=training.evaluate_records
    def evaluate(*a,**kw):
        result=original_evaluate(*a,**kw)
        flag['requested']=True
        return result
    monkeypatch.setattr(training,'evaluate_records',evaluate)
    @contextmanager
    def stop():yield flag
    original_stop=training.stop_at_boundary
    monkeypatch.setattr(training,'stop_at_boundary',stop)
    result=training.train_soft(backend,cfg,[r],[r],{'toy':True},{'development_only':True})
    assert result['status']=='partial' and result['state']['epoch']==1
    out=tmp_path/'run'
    frozen=[file_hash(out/name) for name in ['best_first1.pt','last_first1.pt']]
    assert not (out/'diagnosis_negatives.json').exists()
    checkpoint=read_checkpoint(out/'latest.pt')
    assert checkpoint['run_info']['config']['name_mode']=='parts'
    assert checkpoint['run_info']['diagnosis_negatives'] is None
    monkeypatch.setattr(training,'stop_at_boundary',original_stop)
    complete=training.train_soft(backend,cfg,[r],[r],{'toy':True},{'development_only':True},resume=out/'latest.pt')
    assert complete['status']=='complete' and complete['state']['epoch']==2
    assert [file_hash(out/name) for name in ['best_first1.pt','last_first1.pt']]==frozen
    for path in (out/'validation_predictions').glob('*/summary.json'):
        metrics=json.loads(path.read_text(encoding='utf-8'))
        assert metrics['qa_pairs']==8 and metrics['prediction_rows']==10
    assert not any('contrastive_loss' in x for x in (out/'training.jsonl').read_text(encoding='utf-8').splitlines())
