import copy
import json
import sys

import pytest
import torch

from test_experiment import setup
from medical_kv.config import ROOT, file_hash, fingerprint
from medical_kv.input_cache import attach_cache, cache_manifest, input_hash

sys.path.insert(0,str(ROOT/'scripts'))
from clean_reduced_private_inputs import clean_record, PHONE
from multiscene_common import annotation, split_name, make_qa
from validate_multiscene_dataset import check_record


def record():
    fields={'name':'Ax Bo','phone':'555-234-5678','sender_account':'123 / 4567890',
            'recipient_account':'987 / 7654321','amount_paid':'7 USD'}
    text='\n'.join(('Sender bank and account: 123 / 4567890','Recipient bank and account: 987 / 7654321',
                    'Phone number: 555-234-5678','Name: Ax Bo','Amount: 7 USD'))
    r={'schema_version':'multiscene_privacy_v1','record_id':'fixture','scene':'finance','record_form':'structured',
       'private_prefix':text,'fields':fields,'related_source_keys':['fixture'],'source_case_id':'fixture',
       'group_id':'fixture','split':'train','augmentation_blocks':[],
       'native_text_length':len(text),'native_text_sha256':fingerprint(text),
       'field_annotations':{f:annotation(text,v,'fixture') for f,v in fields.items()},
       'name_annotation':split_name('Ax Bo')}
    r['source_name_fields']=r['name_annotation']['components']; make_qa(r)
    return r


def test_input_deletion_keeps_other_gold_and_adjusts_all_evidence():
    source=record(); before=copy.deepcopy(source)
    r=clean_record(source,'reduced')
    assert source==before and check_record(r)==[]
    assert r['fields']=={'name':'Ax Bo','amount_paid':'7 USD'}
    for f in ('phone','sender_account','recipient_account'):
        assert before['fields'][f] not in r['private_prefix']
    for f,a in r['field_annotations'].items():
        s,e=a['evidence_span']; assert r['private_prefix'][s:e]==r['fields'][f]
    for q in r['qa']:
        s,e=q['evidence_span']; assert r['private_prefix'][s:e]==q['answer']
    assert not list(PHONE.finditer(r['private_prefix']))


def test_all_numbers_under_a_plural_phone_label_are_removed():
    r=record()
    r['private_prefix'] += '\nPhone Numbers: 650-299-4827, +916770254248\n'
    r['native_text_length']=len(r['private_prefix']); r['native_text_sha256']=fingerprint(r['private_prefix'])
    cleaned=clean_record(r,'reduced')
    assert '650-299-4827' not in cleaned['private_prefix']
    assert '+916770254248' not in cleaned['private_prefix']
    assert not list(PHONE.finditer(cleaned['private_prefix']))


def test_caches_preserve_loss_and_gradient_and_reject_wrong_text_or_tokenizer(setup):
    b,cfg,r=setup
    b.identity['tokenizer_sha256']='fixture'
    cached=copy.deepcopy(r)
    cached['_input_cache']={'input_sha256':input_hash(r['private_prefix']+cfg['separator']),
        'tokenizer_sha256':'fixture','ids':b.ids(r['private_prefix']+cfg['separator'])}
    cfg['split_private_prefill']=True
    a=b.embedding(b.tensor(b.initial_ids()))[0].detach().float().requires_grad_(True)
    c=a.detach().clone().requires_grad_(True)
    loss=b.record_loss(r,vectors=a); expected=torch.autograd.grad(loss,a)[0]
    actual=b.record_loss(cached,vectors=c); gradient=torch.autograd.grad(actual,c)[0]
    torch.testing.assert_close(actual,loss,rtol=0,atol=0)
    torch.testing.assert_close(gradient,expected,rtol=0,atol=0)
    assert b.serialized_ids(cached,b.initial_ids())==b.serialized_ids(r,b.initial_ids())
    cached['private_prefix']+=' changed'
    with pytest.raises(ValueError,match='Cached input'):
        b.private_ids(cached)
    cached['private_prefix']=r['private_prefix']; cached['_input_cache']['tokenizer_sha256']='other'
    with pytest.raises(ValueError,match='Cached input'):
        b.private_ids(cached)


def test_cache_manifest_rejects_stale_data_and_modified_tokens(setup,tmp_path):
    b,cfg,r=setup
    dataset=tmp_path/'dataset'; dataset.mkdir(); cache=tmp_path/'cache'; (cache/'medical').mkdir(parents=True)
    (dataset/'manifest.json').write_text('{}')
    (dataset/'tokenization_report.json').write_text('{"tokenizer_sha256":"fixture"}')
    (dataset/'train.jsonl').write_text('fixture')
    item={'record_id':r['record_id'],'input_sha256':input_hash(r['private_prefix']+cfg['separator']),
          'ids':b.ids(r['private_prefix']+cfg['separator'])}
    path=cache/'medical/train.jsonl'; path.write_text(json.dumps(item)+'\n')
    meta={'cache_kind':'private_input_tokens_v1','dataset_manifest_sha256':file_hash(dataset/'manifest.json'),
          'dataset_split_sha256':{'train':file_hash(dataset/'train.jsonl')},'separator':cfg['separator'],
          'tokenizer_sha256':'fixture','file_sha256':{'medical/train.jsonl':file_hash(path)},'counts':{'medical':{'train':1}}}
    (cache/'manifest.json').write_text(json.dumps(meta))
    cfg.update(dataset_directory=str(dataset),input_cache_directory=str(cache))
    cached=dict(r,scene='medical'); attach_cache(cfg,[cached],'train')
    assert cached['_input_cache']['ids']==item['ids']
    path.write_text('changed')
    with pytest.raises(ValueError,match='cache file changed'):
        cache_manifest(cfg)
    path.write_text(json.dumps(item)+'\n'); (dataset/'manifest.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='cache is stale'):
        cache_manifest(cfg)
