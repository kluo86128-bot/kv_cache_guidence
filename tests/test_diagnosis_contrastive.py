import copy
import json
from pathlib import Path

import pytest
import torch

from test_experiment import setup
from medical_kv.config import ROOT, load_config, file_hash, fingerprint
from medical_kv.data import dataset_identity, load_split
from medical_kv.tasks import grouped_loss, questions_for
from medical_kv.diagnosis_contrastive import (build_bank, bank_index, canonical, persist_bank,
    ranking_loss, KINDS, select_negatives, load_or_build_pool, body_candidates,
    omission_candidates, lexical_candidates, edit_similarity)


def enable(cfg, **overrides):
    options = copy.deepcopy(load_config('configs/experiment_v3_diagnosis_contrastive.json')['diagnosis_contrastive'])
    parent = Path(cfg['output_directory']).parent
    resource = parent/'toy_resources.json'
    if not resource.exists():
        terms=['Cold','Colitis','Collagenosis','Colic','Cholera','Cholecystitis',
               'Warm','Warts','Wheals','Whiplash','Wheezing','Wasting',
               'Failure','Fatigue','Fibrosis','Fever','Fistula','Fissure']
        payload = {'conditions':[{'id':x,'name':x,'aliases':[],'codes':[],'source':'unit fixture'} for x in terms],
                   'body_groups':{'fracture':[{'value':x,'evidence':['unit fixture']} for x in ['bone','knee','forearm']]}}
        resource.write_text(json.dumps(dict(payload,fingerprint=fingerprint(payload))),encoding='utf-8')
    options.update(resources_file=str(resource),pool_directory=str(parent/'pools'))
    options.update(overrides)
    cfg['diagnosis_contrastive']=options
    return cfg


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    cfg=load_config('configs/experiment_v3_diagnosis_contrastive.json')
    cfg['diagnosis_contrastive']['pool_directory']=str(tmp_path_factory.mktemp('real_pools'))
    rows=load_split(cfg,'train')
    return cfg,rows,build_bank(rows,cfg)


def test_four_kind_full_bank_no_padding_and_frozen_positives(prepared):
    cfg,rows,bank=prepared
    original=copy.deepcopy(rows); before=dataset_identity(cfg)
    assert len(bank['entries'])==1000
    assert len(KINDS)==4 and 'qualifier_change' not in KINDS
    assert bank['metadata']['max_negatives_per_record']==4
    assert bank['metadata']['negative_answers']==sum(len(r['negatives']) for r in bank['entries'])
    for record,row in zip(rows,bank['entries']):
        assert row['record_id']==record['record_id'] and row['gold']==record['fields']['diagnosis']
        candidates=[c for pool in row['pools'].values() for c in pool]
        assert all(len(pool)<=5 for pool in row['pools'].values())
        assert len({c['answer'].casefold() for c in candidates})==len(candidates)
        assert all(canonical(c['answer'])!=canonical(row['gold']) for c in candidates)
        assert 1<=len(row['negatives'])<=4
        assert row['negatives']==select_negatives(row,epoch=0)
        lexical=row['pools']['label_confusion']
        assert len(lexical)==5
        assert len({c['negative_focus'] for c in lexical})==5
        assert all(c['focus'][0]==c['negative_focus'][0] for c in lexical)
        assert all(c['above_preferred_similarity']==(c['similarity']>=.7) for c in lexical)
        assert [(c['similarity'],c['answer_similarity']) for c in lexical] == sorted(
            [(c['similarity'],c['answer_similarity']) for c in lexical],reverse=True)
        assert all(c['generated'] for c in lexical)
    assert rows==original and before==dataset_identity(cfg)


def test_shared_body_pools_do_not_require_whole_source_diagnoses(prepared):
    cfg,_,bank=prepared
    resource=json.loads((ROOT/cfg['diagnosis_contrastive']['resources_file']).read_text(encoding='utf-8'))
    rules=json.loads((ROOT/cfg['diagnosis_contrastive']['rules_file']).read_text(encoding='utf-8'))
    assert len(resource['body_groups']['sprain'])==9
    assert len(resource['body_groups']['fracture'])==38
    gold='Sprain of ankle (disorder)'
    choices=list(body_candidates(gold,resource,rules))
    assert len(choices)==8
    assert 'Sprain of knee (disorder)' in {c['answer'] for c in choices}
    assert not any('brain' in c['answer'] or 'ankle'==c['after_site'] for c in choices)
    assert list(body_candidates('Concussion injury of brain (disorder)',resource,rules))==[]
    assert list(body_candidates('Anemia (disorder)',resource,rules))==[]
    a,b=[r for r in bank['entries'] if r['gold']==gold][:2]
    assert a['pools'] is b['pools']  # patients share an immutable diagnosis-level pool in memory
    assert len(list(body_candidates('Otitis media (disorder)',resource,rules)))==1


def test_omissions_delete_whole_spans_and_preserve_negation():
    rules=json.loads((ROOT/'configs/diagnosis_negative_rules.json').read_text(encoding='utf-8'))
    cs=list(omission_candidates('Acute bacterial sinusitis (disorder)',rules))
    assert {c['answer'] for c in cs}=={'Acute sinusitis (disorder)','Bacterial sinusitis (disorder)','Sinusitis (disorder)'}
    assert any(c['component_count']==2 for c in cs)
    cs=list(omission_candidates('Chronic kidney disease stage 1 (disorder)',rules))
    assert any(c['answer']=='Chronic kidney disease (disorder)' for c in cs)
    assert all(not c['answer'].endswith('stage (disorder)') for c in cs)
    cs=list(omission_candidates('Concussion with no loss of consciousness (disorder)',rules))
    assert {c['answer'] for c in cs}=={'Concussion (disorder)'}
    assert all(c['changes'][0]['before'].strip()!='no' for c in cs)


def test_lexical_confusion_uses_spelling_and_excludes_synonyms():
    rules=json.loads((ROOT/'configs/diagnosis_negative_rules.json').read_text(encoding='utf-8'))
    def c(i,name,aliases=(),codes=()):return {'id':i,'name':name,'aliases':list(aliases),'codes':list(codes),'source':'fixture'}
    resource={'conditions':[c('a','Hyperlipidemia'),c('b','Hyperglycemia'),c('c','Hypertension'),
        c('a','High lipids'),c('d','Lipid disorder',aliases=['Hyperlipidemia']),
        c('e','Hyperlipidaemia',codes=[['s','same']]),c('f','Sprain of ankle')]}
    candidates=list(lexical_candidates('Hyperlipidemia (disorder)',{('s','same')},resource,rules,.35))
    assert {c['answer'] for c in candidates}=={'Hyperglycemia (disorder)','Hypertension (disorder)'}
    assert all(c['source']['id'] in {'b','c'} and c['focus']!=c['negative_focus'] for c in candidates)
    assert edit_similarity('hyperlipidemia','hyperglycemia')>.4


def test_lexical_shared_codes_and_generic_omissions_are_not_new_diseases():
    rules=json.loads((ROOT/'configs/diagnosis_negative_rules.json').read_text(encoding='utf-8'))
    resource={'conditions':[
        {'id':'a','name':'Gingivitis','aliases':[],'codes':[['ICD-10-CM','K05.10']]},
        {'id':'b','name':'Gingivosis','aliases':[],'codes':[['ICD-10-CM','K05.10']]}]}
    assert list(lexical_candidates('Gingivitis (disorder)',set(),resource,rules,.35))==[]
    resource={'conditions':[{'id':'a','name':'Carious teeth','aliases':['dental caries'],'codes':[]}]}
    assert list(lexical_candidates('Primary dental caries (disorder)',set(),resource,rules,.35))==[]


def test_lexical_same_initial_generated_scaffold_threshold_and_no_padding():
    rules=json.loads((ROOT/'configs/diagnosis_negative_rules.json').read_text(encoding='utf-8'))
    resource={'conditions':[{'id':str(i),'name':name,'aliases':[],'codes':[]}
        for i,name in enumerate(['Hypotension','Hyperventilation','Yypertension','Hypertension'])]}
    gold='Essential hypertension of kidney (disorder)'
    cs=list(lexical_candidates(gold,set(),resource,rules,.7))
    assert [c['answer'] for c in cs]==['Essential hypotension of kidney (disorder)']
    candidate=cs[0]
    assert candidate['focus']=='hypertension' and candidate['negative_focus']=='hypotension'
    assert candidate['similarity']==pytest.approx(10/12)
    change=candidate['changes'][0]
    assert candidate['answer']==gold[:change['start']]+change['after']+gold[change['end']:]
    assert candidate['generated'] and candidate['source']['name']=='Hypotension'
    assert all(c['source']['name']!='Yypertension' for c in lexical_candidates(gold,set(),resource,rules,.01))
    assert list(lexical_candidates('Fracture of bone (disorder)',set(),resource,rules,.7))==[]


def test_lexical_topk_reports_quality_and_refuses_duplicates_to_fill_count(setup):
    _,cfg,row=setup
    enable(cfg,lexical_preferred_similarity=.99)
    bank=build_bank([row],cfg)
    pool=bank['entries'][0]['pools']['label_confusion']
    assert len(pool)==5 and len({c['negative_focus'] for c in pool})==5
    assert not any(c['above_preferred_similarity'] for c in pool)
    quality=bank['metadata']['lexical_quality'][row['fields']['diagnosis']]
    assert quality['count']==5 and quality['above_preferred_count']==0
    assert quality['minimum_similarity']==min(c['similarity'] for c in pool)
    cfg['diagnosis_contrastive']['lexical_candidates_per_diagnosis']=6
    with pytest.raises(ValueError,match='distinct same-initial lexical candidates'):
        build_bank([row],cfg)


def test_lexical_aliases_require_primary_core_attestation_and_six_is_not_capped(setup):
    _,cfg,row=setup
    enable(cfg,lexical_candidates_per_diagnosis=6)
    path=Path(cfg['diagnosis_contrastive']['resources_file'])
    resource=json.loads(path.read_text(encoding='utf-8'))
    resource['conditions'].append({'id':'rigors','name':'Rigors','aliases':['Chills'],'codes':[]})
    resource['conditions'].append({'id':'chills','name':'Chills','aliases':[],'codes':[]})
    resource['conditions'].append({'id':'fake_alias','name':'Fever','aliases':['Coldd'],'codes':[]})
    payload={k:v for k,v in resource.items() if k!='fingerprint'}
    path.write_text(json.dumps(dict(payload,fingerprint=fingerprint(payload))),encoding='utf-8')
    bank=build_bank([row],cfg)
    pool=bank['entries'][0]['pools']['label_confusion']
    assert len(pool)==6 and len({c['negative_focus'] for c in pool})==6
    alias=next(c for c in pool if c['negative_focus']=='chills')
    assert alias['source_term']=='Chills' and alias['source']['name']=='Rigors'
    assert all(c['negative_focus']!='coldd' for c in pool)
    assert all(c['focus'][0]==c['negative_focus'][0] for c in pool)


def test_lexical_inflections_and_alias_only_misspellings_are_not_new_diseases():
    rules=json.loads((ROOT/'configs/diagnosis_negative_rules.json').read_text(encoding='utf-8'))
    names=['Fracture of bone','Fractured ankle','Failure','Fissure']
    resource={'conditions':[{'id':str(i),'name':name,'aliases':[],'codes':[]} for i,name in enumerate(names)]}
    resource['conditions'].append({'id':'alias','name':'Contusion','aliases':['Fractrued ankle'],'codes':[]})
    candidates=list(lexical_candidates('Fracture of bone (disorder)',set(),resource,rules,0.0))
    assert {c['negative_focus'] for c in candidates}=={'failure','fissure'}


def test_resources_exclude_procedures_and_keep_source_evidence(tmp_path):
    import csv
    from medical_kv.negative_resources import assemble_resources
    source=tmp_path/'conditions.csv'
    with source.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['DESCRIPTION','SYSTEM','CODE'])
        writer.writeheader();writer.writerow({'DESCRIPTION':'Sprain of ankle (disorder)','SYSTEM':'s','CODE':'1'})
    nlm=[{'key_id':'a','primary_name':'Knee sprain','is_procedure':False},
         {'key_id':'p','primary_name':'Procedure','is_procedure':True},
         {'key_id':'b','primary_name':'Streptococcal pharyngits','is_procedure':False}]
    resource=assemble_resources(source,nlm,{'url':'fixture'})
    assert {c['name'] for c in resource['conditions']}=={'Sprain of ankle (disorder)','Knee sprain'}
    assert resource['body_groups']['sprain']==[
        {'value':'ankle','evidence':['Sprain of ankle (disorder)']},
        {'value':'knee','evidence':['Knee sprain']}]
    assert resource['fingerprint']==fingerprint({k:v for k,v in resource.items() if k!='fingerprint'})


def test_spelling_multiple_positions_words_and_operations(prepared):
    import re
    _,_,bank=prepared
    operations,counts=set(),set();multiword=False
    for row in bank['entries']:
        for c in row['pools']['spelling_error']:
            assert re.findall(r'[^A-Za-z]+',row['gold'])==re.findall(r'[^A-Za-z]+',c['answer'])
            assert 1<=c['edit_count']==len(c['changes'])<=3
            operations.update(x['operation'] for x in c['changes']);counts.add(c['edit_count'])
            multiword |= len({x['word_index'] for x in c['changes']})>1
    assert counts=={1,2,3} and multiword
    assert operations=={'substitute','delete','insert','duplicate','transpose','reverse_segment'}


def test_cache_hit_skips_construction_across_training_seeds_epochs_and_lr(prepared,monkeypatch):
    import medical_kv.diagnosis_contrastive as module
    cfg,rows,bank=prepared
    original,reference=load_or_build_pool(rows,cfg)
    def forbidden(*a,**kw):raise AssertionError('Cache hit must never rebuild or read test data/network')
    monkeypatch.setattr(module,'_generate_pool',forbidden)
    import medical_kv.data as data
    monkeypatch.setattr(data,'load_split',forbidden)
    import urllib.request
    monkeypatch.setattr(urllib.request,'urlopen',forbidden)
    changed=copy.deepcopy(cfg);changed.update(seed=42,epochs=2,learning_rate=.001)
    other,ref=load_or_build_pool(rows,changed)
    assert ref==reference and other==original
    bound=build_bank(rows,changed)
    assert bound['metadata']['shared_pool']==bank['metadata']['shared_pool']
    assert bound['metadata']['sampling_seed']==42 and bound['metadata']['pool_seed']==17
    assert bound['fingerprint']!=bank['fingerprint']


def test_cache_pool_seed_rules_and_positive_changes_are_versioned(setup,monkeypatch):
    import medical_kv.diagnosis_contrastive as module
    _,cfg,row=setup;enable(cfg)
    _,a=load_or_build_pool([row],cfg)
    changed=copy.deepcopy(cfg);changed['diagnosis_contrastive']['pool_seed']=18
    _,b=load_or_build_pool([row],changed)
    assert a['file']!=b['file']
    changed_row=copy.deepcopy(row);changed_row['fields']['diagnosis']='Warm'
    _,c=load_or_build_pool([changed_row],cfg)
    assert c['file']!=a['file']
    rule=Path(cfg['output_directory']).parent/'rules.json'
    rule.write_text((ROOT/'configs/diagnosis_negative_rules.json').read_text(encoding='utf-8')+' ',encoding='utf-8')
    changed=copy.deepcopy(cfg);changed['diagnosis_contrastive']['rules_file']=str(rule)
    _,d=load_or_build_pool([row],changed)
    assert d['file']!=a['file']


def test_pool_and_resource_tampering_are_rejected(setup):
    _,cfg,row=setup;enable(cfg)
    _,ref=load_or_build_pool([row],cfg)
    path=Path(ref['directory'])/ref['file']
    obj=json.loads(path.read_text(encoding='utf-8'));obj['by_diagnosis'][0]['gold']='Changed'
    path.write_text(json.dumps(obj),encoding='utf-8')
    with pytest.raises(ValueError,match='fingerprint'):load_or_build_pool([row],cfg)
    resource=Path(cfg['diagnosis_contrastive']['resources_file'])
    obj=json.loads(resource.read_text(encoding='utf-8'));obj['conditions'][0]['name']='Tamper'
    resource.write_text(json.dumps(obj),encoding='utf-8')
    with pytest.raises(ValueError,match='fingerprint'):load_or_build_pool([row],cfg)


def test_sampling_patient_epoch_rank_and_resume_independent(prepared):
    _,_,bank=prepared
    row=bank['entries'][0];n=len(row['pools']['spelling_error'])
    picks=[next(c['answer'] for c in select_negatives(row,epoch=e) if c['kind']=='spelling_error') for e in range(n)]
    assert len(set(picks))==n==5
    same=[r for r in bank['entries'] if r['gold']==row['gold']]
    assert len({tuple(c['answer'] for c in select_negatives(r)) for r in same})>1
    expected={r['record_id']:select_negatives(r,epoch=21) for r in bank['entries']}
    for rank in range(4):
        for r in reversed(bank['entries'][rank::4]):assert select_negatives(r,epoch=21)==expected[r['record_id']]
    assert select_negatives(json.loads(json.dumps(row)),epoch=99)==select_negatives(row,epoch=99)
    with pytest.raises(ValueError,match='epoch'):select_negatives(row,epoch=-1)


def test_no_test_blacklist_and_only_training_record_binding(setup,monkeypatch):
    import medical_kv.data as module
    _,cfg,row=setup;enable(cfg)
    def forbidden(*a,**kw):raise AssertionError('Negative construction must not read test splits')
    monkeypatch.setattr(module,'load_split',forbidden)
    build_bank([row],cfg)
    invalid=dict(copy.deepcopy(row),split='test')
    with pytest.raises(ValueError,match='training records'):build_bank([invalid],cfg)


def test_bank_persistence_detects_tampering_and_missing_resume_bank(prepared,tmp_path):
    _,_,bank=prepared
    persist_bank(bank,tmp_path);persist_bank(bank,tmp_path,resume=True)
    path=tmp_path/'diagnosis_negatives.json'
    obj=json.loads(path.read_text(encoding='utf-8'));obj['entries'][0]['negatives'][0]['answer']+=' changed'
    path.write_text(json.dumps(obj),encoding='utf-8')
    with pytest.raises(ValueError,match='differs'):persist_bank(bank,tmp_path,resume=True)
    path.unlink()
    with pytest.raises(ValueError,match='original diagnosis_negatives'):persist_bank(bank,tmp_path,resume=True)


def test_hinge_sign_average_and_gradients():
    positive = torch.tensor(1., requires_grad=True)
    negative = torch.tensor([.5, 2., .9, 3., 4., 5.], requires_grad=True)
    loss = ranking_loss(positive, negative, .2)
    assert loss.item() == pytest.approx((.7 + .3)/6)
    loss.backward()
    assert positive.grad.item() == pytest.approx(2/6)
    torch.testing.assert_close(negative.grad, torch.tensor([-1/6, 0., -1/6, 0., 0., 0.]))
    assert ranking_loss(positive.detach(), negative.detach().repeat(2), .2).item() == pytest.approx(loss.item())


@pytest.mark.parametrize('checkpointed,chunk_size', [(False, 4), (True, 1), (True, 2)])
def test_contrastive_loss_and_u_gradient_match_independent_reference(setup, checkpointed, chunk_size):
    b, cfg, r = setup
    cfg['birth_date_mode'] = 'parts'
    enable(cfg, checkpoint_negatives=checkpointed, negative_batch_size=chunk_size)
    negatives = ['Collder', 'Colld', 'Warm', 'Cool']
    initial = b.embedding(b.tensor(b.initial_ids()))[0].detach().clone()
    reference = initial.clone().requires_grad_(True)
    ctx = b.context(r, vectors=reference)
    questions = questions_for(r, cfg)
    losses = torch.stack([b.answer_loss(ctx,q) for q in questions])
    qa = next(q for q in questions if q['field'] == 'diagnosis')
    negative = torch.stack([b.answer_loss(ctx,dict(qa,answer=a)) for a in negatives])
    positive = losses[-1]
    expected_ce = grouped_loss(losses, questions)
    expected = expected_ce + .2/4 * ranking_loss(positive,negative,.2)
    expected.backward()
    prompt = initial.clone().requires_grad_(True)
    actual, metrics = b.record_contrastive_loss(r, negatives, vectors=prompt)
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=3e-6)
    torch.testing.assert_close(prompt.grad, reference.grad, rtol=3e-4, atol=3e-6)
    torch.testing.assert_close(metrics['ce_loss'], expected_ce)
    assert all(p.grad is None for p in b.model.parameters())
    assert torch.isfinite(prompt.grad).all() and prompt.grad.norm() > 0


def test_zero_weight_recovers_original_loss_and_gradient(setup):
    b, cfg, r = setup
    enable(cfg, weight=0.)
    bank = bank_index(build_bank([r], cfg))
    initial = b.embedding(b.tensor(b.initial_ids()))[0].detach().clone()
    p = initial.clone().requires_grad_(True)
    expected = b.record_loss(r, vectors=p)
    expected.backward()
    q = initial.clone().requires_grad_(True)
    actual, _ = b.record_contrastive_loss(r,[n['answer'] for n in bank[r['record_id']]['negatives']],vectors=q)
    actual.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(q.grad, p.grad, rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize('epoch', [0, 9])
def test_contrastive_global_mean_with_empty_rank(setup, epoch):
    from medical_kv.training import local_batch_backward
    b, cfg, r = setup
    enable(cfg)
    records = [dict(copy.deepcopy(r),record_id=str(i)) for i in range(3)]
    for record,gold in zip(records,['Cold','Acute cold','Fracture of bone (disorder)']):
        record['fields']['diagnosis']=gold
        record['qa'][-1]['answer']=gold
        record['private_prefix']=record['private_prefix'].replace('Cold',gold)
    bank = bank_index(build_bank(records,cfg))
    assert len({len(row['negatives']) for row in bank.values()})>1
    initial = b.embedding(b.tensor(b.initial_ids()))[0].detach().clone()
    p = initial.clone().requires_grad_(True)
    expected = sum(b.record_contrastive_loss(row,[n['answer'] for n in select_negatives(bank[row['record_id']],epoch=epoch)],vectors=p)[0]
                   for row in records)/len(records)
    expected.backward()
    accumulated = torch.zeros_like(p)
    values, diagnostics = [], []
    for rank in range(4):
        q = initial.clone().requires_grad_(True)
        values.extend(local_batch_backward(b,q,records,[0,1,2],rank,4,negative_bank=bank,diagnostics=diagnostics,epoch=epoch))
        accumulated += q.grad
    torch.testing.assert_close(accumulated,p.grad,rtol=3e-4,atol=3e-6)
    assert sum(values)/3 == pytest.approx(expected.item(),abs=3e-6)
    assert len(diagnostics)==3


def test_evaluation_remains_six_free_generation_queries(setup, tmp_path, monkeypatch):
    from medical_kv.evaluation import evaluate_records
    b,cfg,r=setup
    cfg['birth_date_mode']='parts'
    enable(cfg)
    def forbidden(*a,**kw):
        raise AssertionError('Evaluation must not score diagnosis negatives')
    monkeypatch.setattr(b,'record_contrastive_loss',forbidden)
    summary,rows=evaluate_records(b,[r],prompt_ids=b.initial_ids(),output=tmp_path/'evaluation')
    assert summary['qa_pairs']==6 and summary['prediction_rows']==len(rows)==7
    assert all('negative' not in row for row in rows)


def test_new_config_changes_only_objective_and_output_directory():
    original=load_config('configs/experiment_v2_birth_parts.json')
    changed=load_config('configs/experiment_v3_diagnosis_contrastive.json')
    assert {k for k in changed if changed[k]!=original[k]} == {'diagnosis_contrastive','output_directory'}
    for key,value in [('weight',float('nan')),('margin',-1),('max_negatives_per_record',5),('negatives_per_record',6),('negative_batch_size',0),('candidates_per_kind',2),('rules_file',''),('resources_file',''),('pool_directory',''),('pool_seed',-1),('lexical_min_similarity',0),('lexical_preferred_similarity',0),('lexical_candidates_per_diagnosis',4),('lexical_candidates_per_diagnosis',True),('excluded_test_datasets',[])]:
        modified=copy.deepcopy(changed)
        modified['diagnosis_contrastive'][key]=value
        with pytest.raises(ValueError,match='diagnosis_contrastive'):
            load_config('configs/experiment_v3_diagnosis_contrastive.json',{'diagnosis_contrastive':modified['diagnosis_contrastive']})


def test_training_logs_saves_bank_and_resumes_with_frozen_first_window(setup,tmp_path,monkeypatch):
    import medical_kv.training as training
    from medical_kv.io import read_checkpoint
    from contextlib import contextmanager
    b,cfg,r=setup
    enable(cfg)
    cfg.update(birth_date_mode='parts',epochs=2,comparison_epochs=1,early_stopping=False)
    out=Path(cfg['output_directory'])
    class NoPlot:
        def __init__(self,*a,**kw):pass
        def update_training(self,*a):pass
        def update_validation(self,*a):pass
        def save(self):pass
    monkeypatch.setattr(training,'LossCurve',NoPlot)
    flag={'requested':False}
    def evaluate(*a,**kw):
        p=kw['output'];p.mkdir(parents=True)
        (p/'predictions.csv').write_text('implementation-check\n',encoding='utf-8')
        (p/'predictions.jsonl').write_text('{}\n',encoding='utf-8')
        flag['requested']=True
        return {'macro_em':.1,'macro_nll':1.},[]
    @contextmanager
    def stop():yield flag
    monkeypatch.setattr(training,'evaluate_records',evaluate)
    original_stop=training.stop_at_boundary
    monkeypatch.setattr(training,'stop_at_boundary',stop)
    result=training.train_soft(b,cfg,[r],[r],{'toy':True},{'development_only':True})
    assert result['status']=='partial' and result['state']['epoch']==1
    assert (out/'diagnosis_negatives.json').exists()
    frozen=file_hash(out/'best_first1.pt')
    log=json.loads((out/'training.jsonl').read_text(encoding='utf-8').splitlines()[0])
    assert log['diagnosis_negatives_per_record']==2
    assert log['diagnosis_max_negatives_per_record']==4
    assert log['diagnosis_negative_pairs_scored']==2
    assert log['diagnosis_negative_sampling_epoch']==1
    assert log['diagnosis_negative_kind_counts']['spelling_error']==1
    assert log['loss']==pytest.approx(log['ce_loss']+.2/4*log['diagnosis_contrastive_loss'],abs=2e-6)
    assert 0<=log['diagnosis_margin_accuracy']<=log['diagnosis_ranking_accuracy']<=1
    checkpoint=read_checkpoint(out/'latest.pt')
    assert checkpoint['run_info']['diagnosis_negatives']['fingerprint']==checkpoint['run_info']['subsets']['diagnosis_negatives_sha256']
    monkeypatch.setattr(training,'stop_at_boundary',original_stop)
    complete=training.train_soft(b,cfg,[r],[r],{'toy':True},{'development_only':True},resume=out/'latest.pt')
    assert complete['status']=='complete' and complete['state']['step']==2
    assert file_hash(out/'best_first1.pt')==frozen
    assert (out/'final.pt').exists()
    logs=[json.loads(l) for l in (out/'training.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [l['diagnosis_negative_sampling_epoch'] for l in logs]==[1,2]
