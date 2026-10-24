"""Package runnable code, frozen data and tokenizer; omit runtimes/weights/source population."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = list((ROOT / 'medical_kv').glob('*.py'))
    files += list((ROOT / 'tests').glob('*.py'))
    files += list((ROOT / 'configs').glob('*.json'))
    files += [ROOT / name for name in ('requirements-experiment.txt',
              'requirements-experiment-lock-local.txt', 'requirements-server.txt', '实验代码使用说明_v3.md', 'scripts/smoke_experiment.py',
              'scripts/package_experiment.py', 'scripts/plot_loss.py', 'scripts/check_parallel.py', 'pytest.ini', 'README.md', 'AGENTS.md')]
    files += [p for p in (ROOT / 'data/dataset_v1').iterdir() if p.is_file()]
    files += [p for p in (ROOT / 'data/dataset_diagnosis_overlap10').iterdir() if p.is_file()]
    files += [ROOT / 'data/name_naturalization_report.json']
    files += list((ROOT / 'tools/tokenizer/Qwen3-4B').glob('*.json'))
    # Never reintroduce numbered names by packaging legacy prediction CSVs.
    archive = ROOT / 'medical_privacy_prompt_current.zip'
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in hashes:
            z.write(ROOT / name, arcname=name)
        z.writestr('PACKAGE_MANIFEST.json', json.dumps({'files_sha256': hashes,
            'scope': 'experiment code, frozen dataset, tokenizer and tests; no model weights/JVM/venv/raw candidate population',
            'entrypoint': 'python -m medical_kv inspect'}, indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, expected in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
    report = {'archive': archive.name, 'bytes': archive.stat().st_size,
              'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'files': len(hashes), 'integrity': 'PASS'}
    archive.with_name(archive.name + '.sha256.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
