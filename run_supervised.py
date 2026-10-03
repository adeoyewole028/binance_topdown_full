"""One service instance, passive HTTP workers, exactly one supervised owner.

No child is started on import. This launcher never initializes missing financial
state or cancels exchange protection. A failed child stops its peer and exits;
the platform restart re-enters the existing fenced recovery lifecycle.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import os
import signal
import subprocess
import sys
import time
from typing import Any

from runtime_environment import load_launcher_environment, parse_boolean
from state_store import StateStore


def supervise(commands: Sequence[Sequence[str]], *, should_run: Callable[[], bool],
              monitor: Callable[[], None] | None = None,
              on_failure: Callable[[], None] | None = None) -> int:
    children: list[subprocess.Popen[bytes]] = []
    failed = False
    try:
        for command in commands:
            children.append(subprocess.Popen(list(command), start_new_session=os.name != 'nt'))
        while should_run():
            if any(child.poll() is not None for child in children):
                failed = True
                return 1
            if monitor is not None:
                monitor()
            time.sleep(1)
        return 0
    except Exception:
        failed = True
        raise
    finally:
        # Stop HTTP intake and automation together; retained provider-side stops
        # are untouched. Termination cannot turn uncertain effects into retries.
        for child in children:
            if os.name != 'nt':
                try:
                    getattr(os, 'killpg')(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif child.poll() is None:
                child.terminate()
        if failed and on_failure is not None:
            try:
                on_failure()
            except Exception:
                print('{"event":"service_child_failed","alert_delivery":"unavailable"}', flush=True)
        deadline = time.monotonic() + 30
        for child in children:
            try:
                child.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                if os.name == 'nt':
                    child.kill()
                else:
                    try:
                        getattr(os, 'killpg')(child.pid, getattr(signal, 'SIGKILL'))
                    except ProcessLookupError:
                        pass
                child.wait(timeout=5)
        if os.name != 'nt':
            # A dead master cannot reap or wait for its surviving descendants.
            # Retain the original group IDs and clean their entire groups.
            remaining = [child.pid for child in children]
            while remaining and time.monotonic() < deadline:
                alive = []
                for pid in remaining:
                    try:
                        getattr(os, 'killpg')(pid, 0)
                        alive.append(pid)
                    except ProcessLookupError:
                        pass
                remaining = alive
                if remaining:
                    time.sleep(0.1)
            for pid in remaining:
                try:
                    getattr(os, 'killpg')(pid, getattr(signal, 'SIGKILL'))
                except ProcessLookupError:
                    pass


def main() -> int:
    runtime = load_launcher_environment()
    if runtime.development or os.name == 'nt':
        raise ValueError('The supervised production package requires Linux production runtime')
    os.umask(0o077)
    store = StateStore(runtime.data_directory, os.environ.get('LOCAL_ACCOUNT_ID', 'operator-account'),
                       runtime.venue, runtime.mode)
    store.validate()  # Runtime mounted disk only; no automatic init/migration/repair.
    os.chmod(runtime.data_directory, 0o700)
    running = True
    def stop(signum: int, frame: Any) -> None:
        nonlocal running
        running = False
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    worker = [sys.executable, 'run_worker.py', '--scan', '--operations']
    if parse_boolean(os.environ.get('AUTOMATION_ENABLED', 'false')):
        worker.append('--automate')
    port = int(os.environ.get('PORT', '10000'))
    if not 1 <= port <= 65535:
        raise ValueError('Invalid HTTP port')
    web = [sys.executable, '-m', 'gunicorn', '-c', 'gunicorn.conf.py', '-w', '2',
           '-k', 'gthread', '-b', f'0.0.0.0:{port}', 'wsgi:app']
    from operational_monitor import Monitor
    monitor = Monitor(store)
    monitor.step()  # Verify mounted state/backup before admitting HTTP or execution.
    return supervise([worker, web], should_run=lambda: running, monitor=monitor.step,
                     on_failure=lambda: monitor.step(failure=True))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        print('Supervised service unavailable; state retained for recovery', file=sys.stderr)
        raise SystemExit(1) from None
