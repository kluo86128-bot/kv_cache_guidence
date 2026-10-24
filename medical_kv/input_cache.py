"""Validated token caches for fixed private input; model KV remains computed at runtime."""
import hashlib
import json
from .config import file_hash, resolve


def input_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def cache_manifest(cfg):
    directory=resolve(cfg['input_cache_directory'])
    meta=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if meta.get('cache_kind') != 'private_input_tokens_v1':
        raise ValueError('Unknown input cache format')
    dataset=resolve(cfg['dataset_directory'])
    if (meta['dataset_manifest_sha256'] != file_hash(dataset/'manifest.json')
            or meta['separator'] != cfg['separator']):
        raise ValueError('Input token cache is stale: rebuild it for the current data/separator')
    tokenization=json.loads((dataset/'tokenization_report.json').read_text(encoding='utf-8'))
    if meta['tokenizer_sha256'] != tokenization['tokenizer_sha256']:
        raise ValueError('Input cache tokenizer mismatch')
    for split,sha in meta['dataset_split_sha256'].items():
        if file_hash(dataset/(split+'.jsonl')) != sha:
            raise ValueError('Input token cache dataset split changed')
    for name,sha in meta['file_sha256'].items():
        if file_hash(directory/name) != sha:
            raise ValueError('Input token cache file changed: '+name)
    return meta


def attach_cache(cfg,records,split):
    if not cfg.get('input_cache_directory'):
        return
    meta=cache_manifest(cfg)
    directory=resolve(cfg['input_cache_directory'])
    by_scene={}
    for scene in {r['scene'] for r in records}:
        items={}
        for line in (directory/scene/(split+'.jsonl')).open(encoding='utf-8'):
            row=json.loads(line)
            if row['record_id'] in items:
                raise ValueError('Duplicate input-cache record')
            items[row['record_id']]=row
        if len(items)!=meta['counts'][scene][split]:
            raise ValueError('Input cache record count mismatch')
        by_scene[scene]=items
    for r in records:
        item=by_scene[r['scene']].get(r['record_id'])
        if (item is None or item['input_sha256'] != input_hash(r['private_prefix']+cfg['separator'])
                or not item['ids'] or any(type(i) is not int or i<0 for i in item['ids'])):
            raise ValueError('Missing/changed cached private input: '+r['record_id'])
        r['_input_cache']=dict(item,tokenizer_sha256=meta['tokenizer_sha256'])
