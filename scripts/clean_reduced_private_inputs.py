"""Delete the three excluded input fields, retaining auditable source spans and split IDs."""
import argparse
import collections
import copy
import json
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from build_reduced_multiscene_dataset import EXCLUDED, reduce_record
from multiscene_common import ROOT, file_hash, fingerprint, write_json

PHONE = re.compile(r'(?i)(?:\b(?:phone|telephone|mobile|cell)(?:[ _-]*(?:numbers?|no\.?))?\b|\btel\.?\b)'
                   r'[^\n\d+]{0,20}(\+?[\d(][\d() .-]{7,}\d)')
LABEL = re.compile(r'(?i)(?:sender bank and account|recipient bank and account|phone(?: number)?|telephone(?: number)?|mobile(?: number)?|tel\.?)'
                   r'[^A-Za-z0-9\n]{0,8}(?:(?:is|at)\s+)?[^A-Za-z0-9\n]{0,8}$')


def clean_record(original, dataset_id):
    record = reduce_record(original, dataset_id)
    text = original['private_prefix']
    intervals = []
    candidates = []
    for field in EXCLUDED:
        if field in original['fields']:
            candidates += [(m.start(), m.end(), field) for m in re.finditer(re.escape(original['fields'][field]), text)]
    for match in PHONE.finditer(text):
        if 7 <= len(re.sub(r'\D', '', match.group(1))) <= 15:
            candidates.append((match.start(1), match.end(1), 'phone'))
            # Multiple numbers can share a single Phone Numbers label.
            tail=match.end(1)
            while extra:=re.match(r'\s*(?:,|;|/|and|or)\s*(\+?[\d(][\d() .-]{7,}\d)',text[tail:]):
                value=extra.group(1)
                if not 7 <= len(re.sub(r'\D','',value)) <= 15:break
                candidates.append((tail+extra.start(1),tail+extra.end(1),'phone'))
                tail+=extra.end(1)
    # These source labels point to phone text, not identity documents.
    corrected = []
    for start, end, field in candidates:
        if field != 'phone':
            continue
        a = record['field_annotations'].get('identity_number')
        if a and a['evidence_span'] == [start, end]:
            corrected.append('identity_number')
            record['fields'].pop('identity_number')
            record['field_annotations'].pop('identity_number')
            for key in ('qa','training_qa','evaluation_qa'):
                record[key] = [q for q in record[key] if q.get('group',q['field']) != 'identity_number']
    protected = [a['evidence_span'] for a in record['field_annotations'].values()]
    def overlap(start, end):
        return any(max(start,s) < min(end,e) for s,e in protected)
    for start, end, field in candidates:
        if overlap(start,end):
            raise ValueError(f'Input removal overlaps a retained target: {record["record_id"]}/{field}')
        block = next((b for b in original['augmentation_blocks']
                      if b['span'][0] <= start < end <= b['span'][1]
                      and not overlap(*b['span'])), None)
        if block:
            start,end = block['span']
        else:
            line_start = text.rfind('\n',0,start)+1
            line_end = text.find('\n',end)
            line_end = len(text) if line_end < 0 else line_end+1
            before, after = text[line_start:start], text[end:line_end]
            phone_line=re.sub(r'(?i)^[\s*|\-]*(?:phone|telephone|mobile|cell)(?:[ _-]*(?:numbers?|no\.?))?[^\w\d]*','',text[line_start:line_end])
            phone_line=re.sub(r'(?i)\b(?:and|or)\b','',phone_line)
            if field=='phone' and not re.search(r'[A-Za-z]',phone_line) and not overlap(line_start,line_end):
                intervals.append([line_start,line_end,{field}]); continue
            label = LABEL.search(before)
            if label:
                start = line_start + label.start()
                if not re.sub(r'[\s*|:\-]', '', before[:label.start()]) and not re.sub(r'[\s*|.\-]', '', after):
                    start,end = line_start,line_end
                else:
                    if text[end:end+2] == '. ':
                        end += 2
                    elif text[end:end+1] == '.':
                        end += 1
            if overlap(start,end):
                raise ValueError('Expanded input edit overlaps retained evidence')
        intervals.append([start,end,{field}])
    merged = []
    for start,end,fields in sorted(intervals,key=lambda x:(x[0],x[1])):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1],end); merged[-1][2].update(fields)
        else:
            merged.append([start,end,set(fields)])
    def mapped(position):
        return position-sum(max(0,min(position,end)-start) for start,end,_ in merged if start<position)
    pieces=[]; cursor=0
    for start,end,_ in merged:
        pieces.append(text[cursor:start]); cursor=end
    pieces.append(text[cursor:])
    record['private_prefix']=''.join(pieces)
    for field,a in record['field_annotations'].items():
        s,e=a['evidence_span']; a['evidence_span']=[mapped(s),mapped(e)]
        assert record['private_prefix'][mapped(s):mapped(e)]==record['fields'][field]
    for key in ('qa','training_qa','evaluation_qa'):
        for q in record[key]:
            if 'evidence_span' in q:
                q['evidence_span']=[mapped(p) for p in q['evidence_span']]
    blocks=[]
    for b in record['augmentation_blocks']:
        b['span']=[mapped(p) for p in b['span']]
        if b['span'][0]<b['span'][1]:blocks.append(b)
    record['augmentation_blocks']=blocks
    native_length=original['native_text_length']
    record['native_text_length']=mapped(native_length)
    record['native_text_sha256']=fingerprint(record['private_prefix'][:record['native_text_length']])
    record['native_target_count']=sum(a['origin']=='native' for a in record['field_annotations'].values())
    record['input_sanitization']={
        'version':'remove_phone_sender_recipient_v1', 'source_private_prefix_sha256':fingerprint(text),
        'original_native_text_sha256':original['native_text_sha256'], 'original_native_text_length':native_length,
        'removed_original_spans':[{'span':[s,e],'fields':sorted(fs)} for s,e,fs in merged],
        'corrected_source_target_labels':sorted(set(corrected))}
    return record


def build(rebuild=False):
    source=ROOT/'data/dataset_multiscene_v1'
    target=ROOT/'data/dataset_multiscene_v2_reduced'
    manifest=json.loads((target/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('input_revision')=='private_fields_removed_v1' and not rebuild:
        raise ValueError('Dataset inputs are already cleaned; refusing an accidental repeated rebuild')
    for folder in (source,target):
        hashes=json.loads((folder/'artifact_hashes.json').read_text(encoding='utf-8'))
        for name,expected in hashes.items():
            if file_hash(folder/name)!=expected:raise ValueError(f'Changed artifact: {folder/name}')
    stage=ROOT/('data/_clean_multiscene_v2_staging_'+uuid.uuid4().hex[:8])
    if stage.exists():raise FileExistsError(f'Staging already exists: {stage}')
    shutil.copytree(target,stage)
    stats=collections.defaultdict(lambda:collections.Counter())
    corrections=[]
    for split in ('train','validation','test'):
        cell_streams={}
        with (stage/(split+'.jsonl')).open('w',encoding='utf-8',newline='\n') as out:
            try:
                for line in (source/(split+'.jsonl')).open(encoding='utf-8'):
                    original=json.loads(line); r=clean_record(original,manifest['dataset_id'])
                    cell=(r['scene'],r['record_form'])
                    if cell not in cell_streams:
                        cell_streams[cell]=(stage/cell[0]/cell[1]/(split+'.jsonl')).open('w',encoding='utf-8',newline='\n')
                    encoded=json.dumps(r,ensure_ascii=False)+'\n'
                    out.write(encoded); cell_streams[cell].write(encoded)
                    s=stats[r['scene']]; s['records']+=1
                    s['changed_inputs']+=r['private_prefix']!=original['private_prefix']
                    s['removed_characters']+=len(original['private_prefix'])-len(r['private_prefix'])
                    if r['input_sanitization']['corrected_source_target_labels']:corrections.append(r['record_id'])
            finally:
                for stream in cell_streams.values():stream.close()
    manifest.update(input_revision='private_fields_removed_v1',prepared_date='2026-10-24')
    manifest['target_policy'].update(private_prefix_unchanged=False,removed_input_fields=sorted(EXCLUDED),
        input_edit_scope='Delete known excluded fields and explicitly labelled phone values; retain original source snapshots as provenance.')
    manifest['reproduction_code_sha256']['scripts/clean_reduced_private_inputs.py']=file_hash(Path(__file__))
    write_json(stage/'manifest.json',manifest)
    cfg=json.loads((stage/'build_config.json').read_text(encoding='utf-8'))
    cfg['input_revision']=manifest['input_revision']; cfg['input_field_removal']=sorted(EXCLUDED)
    write_json(stage/'build_config.json',cfg)
    write_json(stage/'input_sanitization_report.json',{'status':'PASS','by_scene':dict(stats),
        'corrected_phone_as_identity_records':corrections,'kept_original_record_ids_splits_groups':True,
        'source_archive':'selected_native_sources.jsonl preserves publisher text; it is not model input.'})
    subprocess.run([sys.executable,str(ROOT/'scripts/audit_multiscene_tokens.py'),'--directory',str(stage)],cwd=ROOT,check=True)
    # Publish only a validated complete staging build, directly over the existing version.
    for path in sorted(stage.rglob('*')):
        if path.is_file():
            destination=target/path.relative_to(stage); destination.parent.mkdir(parents=True,exist_ok=True)
            path.replace(destination)
    print(json.dumps({'status':'complete','directory':str(target),'stats':dict(stats),'corrected_labels':corrections}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild',action='store_true',help='Re-derive cleaned inputs from the frozen v1 parent')
    build(parser.parse_args().rebuild)
