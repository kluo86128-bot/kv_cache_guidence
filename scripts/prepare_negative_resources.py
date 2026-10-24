"""Explicit one-time online snapshot preparation; never imported by training."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medical_kv.config import resolve
from medical_kv.io import write_json
from medical_kv.negative_resources import assemble_resources


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='https://clinicaltables.nlm.nih.gov/ctss-downloads/cond_proc_download-2026-10-01.json.zip')
    p.add_argument('--source', default='data/source/candidates_5000/csv/conditions.csv')
    p.add_argument('--output', default='data/negative_resources/diagnosis_v1.json')
    args = p.parse_args()
    out = resolve(args.output)
    if out.exists():
        raise ValueError('Resource snapshot already exists; use another version/path, never overwrite it')
    with urllib.request.urlopen(args.url, timeout=60) as response:
        raw = response.read()
    archive = zipfile.ZipFile(io.BytesIO(raw))
    names = [n for n in archive.namelist() if n.endswith('.json')]
    if len(names) != 1:
        raise ValueError('Expected exactly one NLM terminology JSON member')
    nlm = json.loads(archive.read(names[0]))
    resource = assemble_resources(resolve(args.source), nlm, {
        'url': args.url, 'archive_sha256': hashlib.sha256(raw).hexdigest(),
        'archive_member': names[0], 'source_records': len(nlm)})
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, resource)
    print(json.dumps({'path': str(out), 'fingerprint': resource['fingerprint'], **resource['metadata']}, indent=2))


if __name__ == '__main__':
    main()
