"""Lightweight file-based pause control, including a bridge for an old trainer."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import time
import uuid
from filelock import FileLock, Timeout


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def lock_held(root):
    try:
        with FileLock(str(Path(root) / '.run.lock'), timeout=0):
            return False
    except Timeout:
        return True


def process_start(pid, proc_root=Path('/proc')):
    try:
        # The command name may contain spaces and parentheses.
        return (proc_root / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


class PauseController:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.request_path = self.root / 'pause_request.json'
        self.state_path = self.root / 'pause_state.json'
        self.process_path = self.root / 'training_control.json'

    def requested(self):
        return self.request_path.is_file()

    def begin(self, *, resume, step, world_size):
        if resume:
            self.request_path.unlink(missing_ok=True)
        write_json(self.process_path, {'schema_version': 1, 'status': 'running',
            'pid': os.getpid(), 'host': socket.gethostname(), 'start_ticks': process_start(os.getpid()),
            'world_size': world_size, 'step_at_start': step, 'started_at_utc': now()})

    def finish(self, status, state, *, reason):
        payload = {'status': status, 'reason': reason, 'epoch': state['epoch'],
                   'cursor': state['cursor'], 'step': state['step'], 'at_utc': now()}
        if status == 'paused':
            request = read_json(self.request_path) if self.requested() else {}
            write_json(self.state_path, {**request, **payload, 'checkpoint': str(self.root / 'latest.pt')})
            self.request_path.unlink(missing_ok=True)
        if self.process_path.is_file():
            write_json(self.process_path, {**read_json(self.process_path), **payload})


def legacy_workers(root, project, proc_root=Path('/proc')):
    """Find only actual train-soft workers for this exact directory, not torchrun."""
    result = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = (directory / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            if '--gpus' in argv or any(x.startswith('--gpus=') for x in argv):
                continue  # launcher: signalling it would bypass safe checkpointing
            index = argv.index('-m')
            if argv[index + 1:index + 3] != ['medical_kv', 'train-soft']:
                continue
            output = argv[argv.index('--output-directory') + 1]
            cwd = (directory / 'cwd').resolve()
            destination = Path(output) if Path(output).is_absolute() else Path(project) / output
            if cwd != Path(project).resolve() or destination.resolve() != Path(root).resolve():
                continue
            pid = int(directory.name)
            start = process_start(pid, proc_root)
            if start:
                result.append((pid, start))
        except (OSError, ValueError, IndexError, UnicodeError):
            continue
    return result


def request_pause(root, project):
    controller = PauseController(root)
    if not (controller.root / 'run.json').is_file() or not lock_held(controller.root):
        raise RuntimeError('No active training holds this run lock; nothing to pause')
    process = read_json(controller.process_path) if controller.process_path.is_file() else {}
    current = (process.get('status') == 'running' and process.get('host') == socket.gethostname()
               and process.get('start_ticks') is not None
               and process_start(process.get('pid')) == process['start_ticks'])
    workers = []
    if not current:
        if not Path('/proc').is_dir():
            raise RuntimeError('Legacy trainer pause bridge requires the Linux server')
        workers = legacy_workers(root, project)
        if not workers:
            raise RuntimeError('Could not verify this run\'s live train-soft workers; no signals sent')
    request = {'schema_version': 1, 'request_id': uuid.uuid4().hex, 'requested_at_utc': now(),
               'mechanism': 'request_file' if current else 'legacy_safe_stop_signal',
               'worker_pids': [pid for pid, _ in workers]}
    write_json(controller.request_path, request)
    # An already imported old trainer cannot read newly added pause code. Its
    # SIGTERM handler performs the same save-at-step-boundary operation.
    for pid, start in workers:
        if process_start(pid) != start:
            raise RuntimeError('Training worker changed during pause; refusing to signal a reused PID')
        os.kill(pid, signal.SIGTERM)
    return request


def training_status(root):
    root = Path(root)
    if not (root / 'run.json').is_file():
        raise FileNotFoundError(f'Not a training run: {root}')
    running = lock_held(root)
    summary = read_json(root / 'run_summary.json') if (root / 'run_summary.json').is_file() else {}
    pending = (root / 'pause_request.json').is_file()
    status = 'pause_requested' if running and pending else 'running' if running else summary.get('status', 'stopped')
    if not running and summary.get('stop_reason') in ('paused', 'interrupted'):
        status = 'paused'
    if not running:
        process_path = root / 'training_control.json'
        failure = root / 'failure.json'
        summary_path = root / 'run_summary.json'
        if ((process_path.is_file() and read_json(process_path).get('status') == 'failed')
                or (failure.is_file() and (not summary_path.is_file()
                    or failure.stat().st_mtime_ns > summary_path.stat().st_mtime_ns))):
            status = 'failed'
    return {'status': status, 'running': running, 'checkpoint_exists': (root / 'latest.pt').is_file(),
            'state': summary.get('state'), 'stop_reason': summary.get('stop_reason')}


def wait_paused(root, timeout=600):
    started = time.monotonic()
    while True:
        status = training_status(root)
        if not status['running']:
            if status['status'] != 'paused' or not status['checkpoint_exists']:
                raise RuntimeError(f'Training stopped without an acknowledged safe pause: {status["status"]}')
            controller = PauseController(root)
            if controller.requested() and status.get('state'):
                # Old processes only know the signal handler. Acknowledge their
                # completed save after the run lock has actually been released.
                controller.finish('paused', status['state'], reason=status['stop_reason'])
            return status
        if time.monotonic() - started >= timeout:
            raise TimeoutError('Pause is still pending; use status before starting a recovery')
        time.sleep(1)
