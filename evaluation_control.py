"""Stop a verified evaluator and resume its durable record predictions in the background."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from filelock import FileLock
from medical_kv.control import process_start, write_json
from medical_kv.evaluation_resume import resume_source
from medical_kv.launch import parse_gpus

ROOT = Path(__file__).resolve().parent


def evaluation_processes(directory, project=ROOT, proc_root=Path('/proc')):
    result = []
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        try:
            args = (path / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            module = args.index('medical_kv')
            if args[module + 1] != 'evaluate':
                continue
            if '--output-directory' in args:
                output = args[args.index('--output-directory') + 1]
            else:
                output = next(a.split('=', 1)[1] for a in args if a.startswith('--output-directory='))
            if ((path / 'cwd').resolve() != Path(project).resolve()
                    or (Path(project) / output).resolve() != Path(directory).resolve()):
                continue
            pid = int(path.name)
            start = process_start(pid, proc_root)
            if start:
                result.append((pid, start))
        except (OSError, ValueError, IndexError, StopIteration, UnicodeError):
            continue
    return result


def alive(pid, start):
    try:
        state = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        return state != 'Z' and process_start(pid) == start
    except (OSError, IndexError):
        return False


def pause(directory, timeout=60):
    directory = Path(directory).resolve()
    resume_source(directory)  # Verify the snapshot before stopping anything.
    if os.name != 'posix' or not Path('/proc').is_dir():
        raise RuntimeError('Evaluator process control runs on the Linux server')
    workers = evaluation_processes(directory, ROOT)
    for pid, start in workers:
        if alive(pid, start):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + timeout
    while any(alive(pid, start) for pid, start in workers):
        if time.monotonic() >= deadline:
            raise TimeoutError('Evaluator is still stopping; no recovery was started')
        time.sleep(.25)
    # Refuse to read journals if any verified worker is still writing.
    if evaluation_processes(directory, ROOT):
        raise RuntimeError('Evaluator workers are still active; no recovery was started')
    return {'status': 'stopped', 'stopped_pids': [pid for pid, _ in workers],
            'evaluation_directory': str(directory)}


def recover(directory, gpus, timeout=60):
    directory = Path(directory).resolve()
    ids = parse_gpus(','.join(gpus).replace(' ', ''))
    resume_source(directory)
    with FileLock(str(directory / '.evaluation_recovery.lock'), timeout=0):
        state_file = directory / 'evaluation_recovery_process.json'
        if state_file.is_file():
            previous = json.loads(state_file.read_text(encoding='utf-8'))
            if previous.get('start_ticks') and alive(previous['pid'], previous['start_ticks']):
                raise RuntimeError('A recovery from this evaluation is already running')
        stopped = pause(directory, timeout)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        output = directory.parent / (directory.name + '_recover_' + stamp)
        log = ROOT / 'logs' / ('recover_' + directory.name + '_' + stamp + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', '-m', 'medical_kv', 'evaluate',
                   '--resume-evaluation', str(directory), '--gpus', ','.join(map(str, ids)),
                   '--output-directory', str(output)]
        with log.open('x', encoding='utf-8') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        state = {'pid': child.pid, 'start_ticks': process_start(child.pid), 'gpus': ids,
                 'log': str(log), 'output_directory': str(output), **stopped, 'status': 'launched'}
        write_json(state_file, state)
        return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['pause', 'recover'])
    parser.add_argument('--evaluation-directory', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+')
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args(argv)
    if args.command == 'recover' and not args.gpus:
        parser.error('recover requires --gpus, e.g. --gpus 0 1 2')
    try:
        result = recover(args.evaluation_directory, args.gpus, args.timeout) if args.command == 'recover' else pause(args.evaluation_directory, args.timeout)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Evaluation control: {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
