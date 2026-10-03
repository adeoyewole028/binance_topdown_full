"""BT14 recovery through the real store and authenticated HTTP boundary."""
from pathlib import Path
import json
import time
import sqlite3
from typing import Any
from unittest.mock import patch

import pytest

from tests.legacy_sandbox import LegacySandbox


def test_scheduled_backup_restores_confirmed_financial_provenance(tmp_path: Path) -> None:
    from operations_recovery import BackupSchedule
    from state_admin import restore_state
    with LegacySandbox(tmp_path) as sandbox:
        sandbox.seed_price('BTC/USDT', 100)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '2'})
        before = sandbox.client.get('/api/status').get_json()
        schedule = BackupSchedule(sandbox.module.state_store)
        stamp = time.time()
        backup = schedule.step(stamp)
        assert backup is not None and schedule.step(stamp + 869) is None
        restored = tmp_path / 'restored'
        started = time.monotonic()
        restore_state(sandbox.module.state_store, backup, restored)
        with LegacySandbox(restored) as recovered:
            after = recovered.client.get('/api/status').get_json()
            assert after['positions'] == before['positions']
            assert after['operations'] == before['operations']
            assert after['fills'] == before['fills']
            assert after['paper_account']['balances'] == before['paper_account']['balances']
        assert time.monotonic() - started < 3600
        assert schedule.step(stamp + 870) is not None


def test_retention_only_removes_verified_owned_expired_backups(tmp_path: Path) -> None:
    from operations_recovery import BackupSchedule
    with LegacySandbox(tmp_path) as sandbox:
        schedule = BackupSchedule(sandbox.module.state_store)
        stamp = time.time()
        first = schedule.step(stamp)
        assert first is not None
        unrelated = schedule.directory / 'operator-notes'
        unrelated.mkdir()
        (unrelated / 'keep').write_text('retain')
        incomplete = schedule.directory / '.incomplete-test'
        incomplete.mkdir()
        second = schedule.step(stamp + 7 * 86400)
        assert second is not None and first.exists()
        schedule.step(stamp + 7 * 86400 + 900)
        assert not first.exists() and second.exists()
        assert (unrelated / 'keep').read_text() == 'retain'
        assert incomplete.exists()


def test_failed_backup_never_replaces_last_complete_snapshot(tmp_path: Path) -> None:
    from operations_recovery import BackupSchedule
    from state_store import StateUnavailable
    with LegacySandbox(tmp_path) as sandbox:
        schedule = BackupSchedule(sandbox.module.state_store)
        stamp = time.time()
        first = schedule.step(stamp)
        connect = sqlite3.connect
        class DiskFull(sqlite3.Connection):
            def backup(self, target: sqlite3.Connection, **kwargs: Any) -> None:
                raise OSError('disk full')
        def failing_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
            return connect(*args, **kwargs, factory=DiskFull)
        with patch('sqlite3.connect', failing_connect):
            with pytest.raises(StateUnavailable):
                schedule.step(stamp + 900)
        assert first is not None and first.exists()
        assert len(list(schedule.directory.glob('snapshot-*'))) == 1


def test_restart_validates_existing_backup_and_corruption_blocks_reuse(tmp_path: Path) -> None:
    from operations_recovery import BackupSchedule
    from state_store import StateUnavailable
    with LegacySandbox(tmp_path) as sandbox:
        stamp = time.time()
        first = BackupSchedule(sandbox.module.state_store).step(stamp)
        assert first is not None
        assert BackupSchedule(sandbox.module.state_store).step(stamp + 10) is None
        (first / 'trading.sqlite3').write_bytes(b'corrupt')
        with pytest.raises(StateUnavailable):
            BackupSchedule(sandbox.module.state_store).step(stamp + 20)


def test_clock_reversal_and_foreign_backup_never_claim_current_recovery(tmp_path: Path) -> None:
    from operations_recovery import BackupSchedule
    from state_store import StateUnavailable
    with LegacySandbox(tmp_path) as sandbox:
        schedule = BackupSchedule(sandbox.module.state_store)
        stamp = time.time()
        first = schedule.step(stamp)
        assert first is not None
        with pytest.raises(StateUnavailable):
            schedule.step(stamp - 1)
        metadata = first / 'retention.json'
        values = json.loads(metadata.read_text())
        values['store_id'] = 'different-store'
        metadata.write_text(json.dumps(values))
        with pytest.raises(StateUnavailable):
            BackupSchedule(sandbox.module.state_store).step(stamp + 10)


def test_public_health_requires_fresh_matching_owner_and_exposes_no_account_data(tmp_path: Path) -> None:
    from execution_worker import ExecutionWorker
    from operational_health import OwnerMaintenance
    from tests.test_market_runtime import ControlledClock
    clock = ControlledClock()
    with LegacySandbox(tmp_path, authenticated=False) as sandbox:
        sandbox.module.app.config['MARKET_CLOCK'] = clock
        assert sandbox.client.get('/healthz').status_code == 503
        assert sandbox.client.get('/api/operational-health').status_code == 401
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            maintenance = OwnerMaintenance(owner)
            from operations_recovery import BackupSchedule
            BackupSchedule(sandbox.module.state_store).step(clock())
            maintenance.step()
            response = sandbox.client.get('/healthz')
            assert response.status_code == 200
            assert response.get_json() == {'control_ready': True}
            clock.value += 15
            assert sandbox.client.get('/healthz').status_code == 503
            maintenance.step()
            assert sandbox.client.get('/healthz').status_code == 200
        assert sandbox.client.get('/healthz').status_code == 503


def test_maintenance_failure_prevents_owned_loop_from_consuming_queued_intent(tmp_path: Path) -> None:
    from execution_worker import ExecutionWorker
    from operational_health import OwnerMaintenance
    from run_worker import run_owned_loop
    from state_store import StateUnavailable
    with LegacySandbox(tmp_path) as sandbox:
        sandbox.seed_price('BTC/USDT', 100)
        sandbox.client.post('/trade', data={'csrf_token': sandbox.csrf,
            'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'})
        (tmp_path / 'backups').write_text('filesystem unavailable')
        with ExecutionWorker(sandbox.module.state_store) as owner:
            with pytest.raises(StateUnavailable):
                run_owned_loop(owner, once=True, should_run=lambda: True,
                               maintenance=OwnerMaintenance(owner).step)
        after = sandbox.client.get('/api/status').get_json()
        assert after['operations'][0]['state'] == 'queued'
        assert after['fills'] == [] and after['positions'] == []


def test_redacted_operator_metrics_keep_stale_scanner_blocked(tmp_path: Path) -> None:
    from execution_worker import ExecutionWorker
    from operational_health import OwnerMaintenance
    from market_runtime import MarketRuntime
    from tests.test_market_runtime import ControlledClock, MarketFixture
    clock = ControlledClock()
    with LegacySandbox(tmp_path) as sandbox:
        sandbox.module.app.config['MARKET_CLOCK'] = clock
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            MarketRuntime(owner, MarketFixture(clock()), clock=clock).scan_once()
            from operations_recovery import BackupSchedule
            BackupSchedule(sandbox.module.state_store).step(clock())
            OwnerMaintenance(owner).step()
            report = sandbox.client.get('/api/operational-health').get_json()
            assert report['entry_ready'] is True and report['control_ready'] is True
            clock.value += 601
            report = sandbox.client.get('/api/operational-health').get_json()
            assert report['entry_ready'] is False and report['control_ready'] is False
            assert 'password' not in json.dumps(report) and 'balance' not in json.dumps(report)


def test_monitor_persists_redacted_incidents_without_external_delivery(tmp_path: Path) -> None:
    from operational_monitor import Monitor
    with LegacySandbox(tmp_path) as sandbox:
        monitor = Monitor(sandbox.module.state_store)
        monitor.step()
        first = json.loads((tmp_path / 'operations-alert.json').read_text())
        assert first['delivered'] is False
        assert 'owner_unavailable' in first['conditions']
        monitor.next_check = 0
        monitor.step()
        assert json.loads((tmp_path / 'operations-alert.json').read_text())['event_id'] == first['event_id']
        assert sandbox.network_attempts == 0


def test_supervisor_stops_peer_when_one_external_child_fails(tmp_path: Path) -> None:
    import subprocess
    import sys
    from tests.legacy_sandbox import ROOT
    # A separate guarded harness runs actual child processes; it never launches
    # production providers, HTTP services, or inherited credentials.
    script = '''
import sys
from run_supervised import supervise
result=supervise([[sys.executable,'-c','raise SystemExit(2)'],
                  [sys.executable,'-c','import time;time.sleep(60)']],should_run=lambda:True)
assert result == 1
print('Peer stopped')
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'Peer stopped'


def test_supervisor_stops_all_children_when_operational_persistence_fails() -> None:
    import subprocess
    import sys
    from tests.legacy_sandbox import ROOT
    script = '''
import sys
from run_supervised import supervise
from state_store import StateUnavailable
def fail():raise StateUnavailable('storage unavailable')
try:
    supervise([[sys.executable,'-c','import time;time.sleep(60)']],
              should_run=lambda:True,monitor=fail)
except StateUnavailable:pass
else:raise AssertionError('Failure did not stop service')
print('Persistence failure stopped service')
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'Persistence failure stopped service'
