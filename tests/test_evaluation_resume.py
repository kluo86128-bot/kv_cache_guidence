import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_experiment import setup
from test_multiscene_runtime import configure, records, perfect
from medical_kv import evaluation
from medical_kv.config import file_hash
from medical_kv.evaluation_resume import completed_predictions, resume_source, validate_metadata
from medical_kv.io import save_checkpoint, write_json


def metadata(backend, snapshot):
    return {'condition':'soft','split':'test','evaluation_stage':'interim','partial_evaluation':False,
            'checkpoint_run_signature':'fixture','epoch':25,'step':100,'dataset':{'sha':'fixture'},
            'backend':backend.identity,'generation':{'max_new_tokens':2},
            'resolved_config':copy.deepcopy(backend.cfg), 'source_checkpoint_sha256':file_hash(snapshot),
            'parallelism':{'world_size':2,'visible_gpu_ids':'0,2'}}


def prepare(backend, tmp_path, monkeypatch, rows=None):
    configure(backend.cfg)
    monkeypatch.setattr(backend, 'generate_many', perfect)
    rows = records() if rows is None else rows
    snapshot = tmp_path / 'checkpoint.pt'
    save_checkpoint(snapshot, {'prompt':torch.ones(3,32)})
    meta = metadata(backend, snapshot)
    _, answers = evaluation.evaluate_records(backend, rows, prompt_ids=backend.initial_ids())
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'checkpoint.pt').write_bytes(snapshot.read_bytes())
    write_json(source / 'metadata.json', meta)
    return source, meta, answers, rows


def journal(path, rows, tail=b''):
    path.write_bytes(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode()+tail)


def test_legacy_two_rank_journals_reuse_complete_records_and_replay_torn_record(setup, tmp_path, monkeypatch):
    backend, _, _ = setup
    source, meta, answers, rows = prepare(backend,tmp_path,monkeypatch)
    journal(source/'predictions.rank0.partial.jsonl',[r for r in answers if r['record_id']==rows[0]['record_id']])
    tail_rows = [r for r in answers if r['record_id']==rows[2]['record_id']][:2]
    journal(source/'predictions.rank1.partial.jsonl',[r for r in answers if r['record_id']==rows[1]['record_id']]+tail_rows,b'{"record_id":')
    before = {p.name:p.read_bytes() for p in source.iterdir()}
    calls = []
    context = backend.context
    monkeypatch.setattr(backend,'context',lambda record,**kw: calls.append(record['record_id']) or context(record,**kw))
    summary, actual = evaluation.evaluate_records(backend,rows,prompt_ids=backend.initial_ids(),
        output=tmp_path/'recovered',metadata=meta,resume_evaluation=source)
    assert calls == [rows[2]['record_id']]
    assert actual == answers
    assert summary['metadata']['evaluation_resume']['completed_records']==2
    assert summary['metadata']['evaluation_resume']['incomplete_records_replayed']==[rows[2]['record_id']]
    assert {p.name:p.read_bytes() for p in source.iterdir()}==before
    saved=[json.loads(l) for l in (tmp_path/'recovered/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert saved==answers
    assert (tmp_path/'recovered/predictions.csv').exists()


def test_recover_again_after_second_interruption_preserves_previously_reused_records(setup,tmp_path,monkeypatch):
    backend,_,_=setup
    source,meta,answers,rows=prepare(backend,tmp_path,monkeypatch)
    journal(source/'predictions.rank0.partial.jsonl',[r for r in answers if r['record_id']==rows[0]['record_id']])
    calls=[]
    def interrupted(context,qs):
        calls.append(qs)
        if len(calls)==2:
            raise RuntimeError('interrupted fixture')
        return perfect(context,qs)
    monkeypatch.setattr(backend,'generate_many',interrupted)
    first=tmp_path/'first_recovery'
    with pytest.raises(RuntimeError,match='interrupted fixture'):
        evaluation.evaluate_records(backend,rows,prompt_ids=backend.initial_ids(),output=first,metadata=meta,resume_evaluation=source)
    assert (first/'reused_predictions.jsonl').exists()
    monkeypatch.setattr(backend,'generate_many',perfect)
    calls=[]
    context=backend.context
    monkeypatch.setattr(backend,'context',lambda record,**kw:calls.append(record['record_id']) or context(record,**kw))
    _,actual=evaluation.evaluate_records(backend,rows,prompt_ids=backend.initial_ids(),
        output=tmp_path/'second_recovery',metadata=meta,resume_evaluation=first)
    assert calls==[rows[2]['record_id']] and actual==answers


def test_changed_identity_corrupt_lines_and_conflicting_duplicates_are_rejected(setup,tmp_path,monkeypatch):
    backend,_,_=setup
    source,meta,answers,rows=prepare(backend,tmp_path,monkeypatch)
    for key in ('dataset','generation','epoch','checkpoint_run_signature'):
        changed=copy.deepcopy(meta); changed[key]='changed'
        with pytest.raises(ValueError,match='identity mismatch'):
            validate_metadata(meta,changed)
    path=source/'predictions.rank0.partial.jsonl'
    journal(path,answers,b'invalid\n')
    with pytest.raises(ValueError,match='Corrupt prediction'):
        completed_predictions(source,rows,backend.cfg)
    changed=copy.deepcopy(answers[0]); changed['prediction']='different'
    journal(path,answers+[changed])
    with pytest.raises(ValueError,match='Conflicting duplicate'):
        completed_predictions(source,rows,backend.cfg)
    changed=copy.deepcopy(answers); changed[0]['gold']='wrong'
    journal(path,changed)
    with pytest.raises(ValueError,match='question/gold changed'):
        completed_predictions(source,rows,backend.cfg)
    (source/'checkpoint.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='snapshot is missing or changed'):
        resume_source(source)


def test_pending_records_repartition_from_two_to_three_ranks_without_duplicates(setup,tmp_path,monkeypatch):
    backend,_,_=setup
    rows=[]
    for i in range(7):
        record=copy.deepcopy(records()[i%3]); record['record_id']=f'record_{i}'
        rows.append(record)
    source,meta,answers,rows=prepare(backend,tmp_path,monkeypatch,rows)
    complete={r['record_id'] for r in rows[:3]}
    journal(source/'predictions.rank0.partial.jsonl',[a for a in answers if a['record_id'] in {rows[0]['record_id'],rows[2]['record_id']}])
    journal(source/'predictions.rank1.partial.jsonl',[a for a in answers if a['record_id']==rows[1]['record_id']])
    restored,progress=completed_predictions(source,rows,backend.cfg)
    assignments=[[r['record_id'] for r in rows[3:][rank::3]] for rank in range(3)]
    shards=[[a for a in answers if a['record_id'] in set(assignment)] for assignment in assignments]
    shards[0]=restored+shards[0]
    class Runtime:
        active=True; world_size=3
        def __init__(self,device):
            self.rank=current[0]; self.primary=self.rank==0; self.calls=0
        def broadcast(self,value):
            self.calls+=1
            return progress if self.calls==1 else sorted(complete)
        def barrier(self):pass
        def objects(self,value):
            assert value==shards[self.rank]
            return shards
        def maximum(self,value):return value
        def metadata(self):return {'world_size':3,'visible_gpu_ids':'0,1,2','backend':'simulated'}
    monkeypatch.setattr(evaluation,'Runtime',Runtime)
    context=backend.context
    observed=[]
    monkeypatch.setattr(backend,'context',lambda record,**kw:observed.append(record['record_id']) or context(record,**kw))
    current=[0]
    for rank in range(3):
        current[0]=rank; observed.clear()
        summary,actual=evaluation.evaluate_records(backend,rows,prompt_ids=backend.initial_ids(),
            output=tmp_path/'three_rank_recovery',metadata=meta,resume_evaluation=source)
        assert observed==assignments[rank]
        assert actual==answers and summary['records']==7
    assert sorted(x for assignment in assignments for x in assignment)==[r['record_id'] for r in rows[3:]]


def test_evaluation_process_scan_excludes_other_outputs_and_trainers(tmp_path):
    import evaluation_control as control
    project=tmp_path/'project'; project.mkdir()
    proc=tmp_path/'proc'; proc.mkdir()
    target=tmp_path/'evaluation'
    def process(pid,args):
        folder=proc/str(pid); folder.mkdir(); (folder/'cwd').mkdir()
        (folder/'cmdline').write_bytes(('\0'.join(args)+'\0').encode())
        (folder/'stat').write_text(f'{pid} (python) '+' '.join(['S']+['x']*18+['start']))
        return folder/'cwd'
    cwd=process(123,['python','-m','medical_kv','evaluate','--output-directory',str(target)])
    assert control.evaluation_processes(target,cwd,proc)==[(123,'start')]
    process(124,['python','-m','medical_kv','train-soft','--output-directory',str(target)])
    process(125,['python','-m','medical_kv','evaluate','--output-directory',str(tmp_path/'other')])
    assert control.evaluation_processes(target,cwd,proc)==[(123,'start')]


def test_recovery_launches_new_background_directory_on_requested_gpus(tmp_path,monkeypatch):
    import evaluation_control as control
    source=tmp_path/'evaluation'; source.mkdir()
    monkeypatch.setattr(control,'resume_source',lambda p: ({},p/'checkpoint.pt'))
    monkeypatch.setattr(control,'pause',lambda p,t:{'status':'stopped','stopped_pids':[1,2]})
    monkeypatch.setattr(control,'process_start',lambda p:'start')
    monkeypatch.setattr(control,'ROOT',tmp_path)
    calls=[]
    monkeypatch.setattr(control.subprocess,'Popen',lambda args,**kw:calls.append((args,kw)) or SimpleNamespace(pid=123))
    result=control.recover(source,['0','1','2'])
    assert result['gpus']==[0,1,2] and result['status']=='launched'
    assert Path(result['output_directory'])!=source and Path(result['log']).parent==tmp_path/'logs'
    args,options=calls[0]
    assert args[args.index('--resume-evaluation')+1]==str(source)
    assert args[args.index('--gpus')+1]=='0,1,2' and options['start_new_session'] is True
