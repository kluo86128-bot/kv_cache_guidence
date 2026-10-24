"""Rebuild loss_curve.png from an existing soft training directory."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medical_kv.plotting import LossCurve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', required=True)
    args = parser.parse_args()
    directory = Path(args.run_directory)
    if not directory.is_absolute():
        directory = ROOT / directory
    run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    if run['method'] != 'soft':
        parser.error('Loss plotting currently supports soft training only')
    path = LossCurve(directory, implementation_only=run['backend'].get('model') == 'random-tiny-Qwen3').save()
    if path is None:
        parser.error('No training loss records found')
    print(path)


if __name__ == '__main__':
    main()
