"""Shared recovery/ownership acceptance at the approved operator boundary."""
from pathlib import Path

from tests.legacy_sandbox import LegacySandbox
import sqlite3
from typing import Any, Literal
import uuid
from unittest.mock import patch


def test_two_http_workers_read_the_same_committed_configuration(tmp_path: Path) -> None:
    with LegacySandbox(tmp_path) as first:
        with LegacySandbox(tmp_path) as second:
            response = first.post('/update-config', data={'risk': '0.02'})
            assert response.status_code == 302
            assert second.client.get('/api/status').get_json()['config']['risk'] == 0.02


def test_failed_database_commit_keeps_the_previous_configuration(tmp_path: Path) -> None:
    with LegacySandbox(tmp_path) as sandbox:
        class FailingCommit(sqlite3.Connection):
            def commit(self) -> None:
                raise sqlite3.OperationalError('simulated disk-full commit failure')
        connect = sqlite3.connect
        def failing_connect(database: str, *, timeout: float = 5,
                            isolation_level: Literal['DEFERRED', 'EXCLUSIVE', 'IMMEDIATE'] | None = None,
                            uri: bool = False) -> sqlite3.Connection:
            return connect(database, timeout=timeout, isolation_level=isolation_level,
                           uri=uri, factory=FailingCommit)
        with patch('sqlite3.connect', failing_connect):
            assert sandbox.post('/update-config', data={'risk': '0.03'}).status_code == 503
        assert sandbox.client.get('/api/status').get_json()['config']['risk'] == 0.01


def test_missing_required_database_never_becomes_an_empty_account(tmp_path: Path) -> None:
    with LegacySandbox(tmp_path) as sandbox:
        (tmp_path / 'trading.sqlite3').unlink()
        assert sandbox.client.get('/api/status').status_code == 503
        assert sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'}).status_code == 503
        assert sandbox.exchange.holdings == {}
        assert not (tmp_path / 'trading.sqlite3').exists()


def test_order_intake_is_durable_and_http_workers_do_not_dispatch(tmp_path: Path) -> None:
    with LegacySandbox(tmp_path) as sandbox:
        sandbox.seed_price('BTC/USDT', 100.0)
        # Direct HTTP boundary, before running the explicit owner.
        response = sandbox.client.post('/trade', data={
            'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf,
        })
        assert response.status_code == 302
        payload = sandbox.client.get('/api/status').get_json()
        assert payload['open_positions'] == []
        assert len(payload['operations']) == 1
        assert payload['operations'][0]['state'] == 'queued'
        assert sandbox.exchange.holdings == {}


def test_second_execution_owner_cannot_acquire_the_same_account(tmp_path: Path) -> None:
    from execution_owner import ExecutionOwner, OwnershipUnavailable
    import pytest
    with LegacySandbox(tmp_path) as sandbox:
        with ExecutionOwner(sandbox.module.state_store):
            with pytest.raises(OwnershipUnavailable):
                with ExecutionOwner(sandbox.module.state_store):
                    raise AssertionError('A second owner acquired the same account')
        with ExecutionOwner(sandbox.module.state_store) as recovered:
            assert recovered.ownership_confirmed()


def test_explicit_owner_commits_paper_fill_visible_to_both_http_workers(tmp_path: Path) -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(tmp_path) as first:
        with LegacySandbox(tmp_path) as second:
            first.seed_price('BTC/USDT', 100.0)
            first.client.post('/trade', data={
                'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '2', 'csrf_token': first.csrf,
            })
            with ExecutionWorker(first.module.state_store) as worker:
                worker.process_pending()
            for client in (first.client, second.client):
                state = client.get('/api/status').get_json()
                assert state['positions'][0]['qty'] == '2'
                assert state['operations'][0]['state'] == 'filled'


def test_exact_decimal_close_does_not_leave_a_float_rounding_remainder() -> None:
    with LegacySandbox() as sandbox:
        sandbox.seed_price('BTC/USDT', 100.0)
        quantity = '0.123456789123456789'
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': quantity})
        assert sandbox.client.get('/api/status').get_json()['positions'][0]['qty'] == quantity
        sandbox.post('/close', data={'symbol': 'BTC/USDT', 'qty': quantity})
        assert sandbox.client.get('/api/status').get_json()['positions'] == []


def test_queued_paper_orders_follow_intake_order_instead_of_random_identity() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox() as sandbox:
        sandbox.seed_price('BTC/USDT', 100.0)
        identities = [uuid.UUID(int=value) for value in range(8, 0, -1)]
        with patch('uuid.uuid4', side_effect=identities):
            for side in ('buy', 'sell') * 4:
                sandbox.client.post('/trade', data={
                    'symbol': 'BTC/USDT', 'side': side, 'qty': '1', 'csrf_token': sandbox.csrf,
                })
        with ExecutionWorker(sandbox.module.state_store) as worker:
            worker.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['positions'] == []
        assert all(operation['state'] == 'filled' for operation in state['operations'])


def test_mainnet_and_testnet_namespaces_do_not_share_positions_or_config(tmp_path: Path) -> None:
    with LegacySandbox(tmp_path, environment={'BINANCE_VENUE': 'testnet'}) as testnet:
        testnet.seed_price('BTC/USDT', 100.0)
        testnet.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '2'})
        testnet.post('/update-config', data={'risk': '0.02'})
        with LegacySandbox(tmp_path, environment={'BINANCE_VENUE': 'mainnet'}) as mainnet:
            state = mainnet.client.get('/api/status').get_json()
            assert state['venue'] == 'mainnet'
            assert state['positions'] == []
            assert state['config']['risk'] == 0.01
            mainnet.post('/update-config', data={'risk': '0.03', 'namespace': testnet.module.state_store.namespace})
            assert testnet.client.get('/api/status').get_json()['config']['risk'] == 0.02
            assert testnet.client.get('/api/status').get_json()['positions'][0]['qty'] == '2'


def test_legacy_snapshot_stays_quarantined_until_exact_paper_origin_is_confirmed(tmp_path: Path) -> None:
    import json
    from state_admin import quarantine_legacy, confirm_paper_snapshot
    source = tmp_path / 'positions.json'
    source.write_text(json.dumps([
        {'symbol': 'BTC/USDT', 'qty': '0.125', 'entry': '100'},
        {'symbol': 'ETH/USDT', 'qty': '-2', 'entry': '20'},
    ]), encoding='utf-8')
    with LegacySandbox(tmp_path) as sandbox:
        manifest = quarantine_legacy(sandbox.module.state_store, [source])
        assert sandbox.client.get('/api/status').get_json()['positions'] == []
        assert manifest[0]['state'] == 'quarantined'
        checksum = manifest[0]['checksum']
        confirm_paper_snapshot(sandbox.module.state_store, checksum, actor='operator',
                               origin_confirmation=f'paper:{sandbox.module.state_store.namespace}:{checksum}')
        confirm_paper_snapshot(sandbox.module.state_store, checksum, actor='operator',
                               origin_confirmation=f'paper:{sandbox.module.state_store.namespace}:{checksum}')
        state = sandbox.client.get('/api/status').get_json()
        assert state['positions'] == [{'symbol': 'BTC/USDT', 'qty': '0.125', 'entry': '100',
                                      'basis_status': 'legacy-opening; fees/capital/history unknown'}]
        assert state['migration'][0]['imported_count'] == 1
        assert state['migration'][0]['quarantine_count'] == 1
        assert state['operations'] == []  # An opening snapshot is not an invented fill.
        assert quarantine_legacy(sandbox.module.state_store, [source])[0]['imported_count'] == 1


def test_restore_recovers_backup_and_refuses_to_discard_newer_journal(tmp_path: Path) -> None:
    import pytest
    from state_admin import backup_state, restore_state
    from state_store import StateUnavailable
    current = tmp_path
    backup = tmp_path / 'backup'
    restored = tmp_path / 'restored'
    with LegacySandbox(current) as sandbox:
        sandbox.seed_price('BTC/USDT', 100)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '2'})
        backup_state(sandbox.module.state_store, backup)
        restore_state(sandbox.module.state_store, backup, restored)
        with LegacySandbox(restored) as recovered:
            assert recovered.client.get('/api/status').get_json()['positions'][0]['qty'] == '2'
            assert recovered.client.get('/api/status').get_json()['operations'][0]['state'] == 'filled'
        sandbox.post('/close', data={'symbol': 'BTC/USDT', 'qty': '1'})
        with pytest.raises(StateUnavailable, match='newer journal'):
            restore_state(sandbox.module.state_store, backup, tmp_path / 'stale-restore')
        assert sandbox.client.get('/api/status').get_json()['positions'][0]['qty'] == '1'


def test_failed_fill_commit_survives_restart_as_one_recoverable_intent(tmp_path: Path) -> None:
    import pytest
    from execution_worker import ExecutionWorker
    from state_store import StateUnavailable
    with LegacySandbox(tmp_path) as sandbox:
        sandbox.seed_price('BTC/USDT', 100)
        sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '2', 'csrf_token': sandbox.csrf})
        class FailingCommit(sqlite3.Connection):
            def commit(self) -> None:
                raise sqlite3.OperationalError('simulated full disk')
        connect = sqlite3.connect
        def failing_connect(database: str, *, timeout: float = 5,
                            isolation_level: Literal['DEFERRED', 'EXCLUSIVE', 'IMMEDIATE'] | None = None,
                            uri: bool = False) -> sqlite3.Connection:
            return connect(database, timeout=timeout, isolation_level=isolation_level, uri=uri, factory=FailingCommit)
        with ExecutionWorker(sandbox.module.state_store) as worker:
            with patch('sqlite3.connect', failing_connect), pytest.raises(StateUnavailable):
                worker.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['positions'] == []
        assert state['operations'][0]['state'] == 'queued'
    with LegacySandbox(tmp_path) as restarted:
        restarted.run_owner()
        restarted.run_owner()
        state = restarted.client.get('/api/status').get_json()
        assert state['positions'][0]['qty'] == '2'
        assert len(state['operations']) == 1
        assert state['operations'][0]['state'] == 'filled'


def test_real_process_owner_excludes_contender_and_recovers_after_process_death(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys
    # No child reads .env or constructs a provider. Explicit disposable paths only.
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('BINANCE_', 'OPERATOR_', 'ALLOW_', 'TRADING_'))
                   and key not in {'API_KEY', 'API_SECRET', 'FLASK_SECRET_KEY', 'DATA_DIR'}}
    command = [sys.executable, '-m', 'tests.owner_process', str(tmp_path)]
    with LegacySandbox(tmp_path) as sandbox:
        sandbox.seed_price('BTC/USDT', 100)
        sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '3', 'csrf_token': sandbox.csrf})
        owner = subprocess.Popen([*command, 'hold'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, env=environment)
        try:
            assert owner.stdout is not None
            assert owner.stdout.readline().strip() == 'owned'
            with (tmp_path / 'contender-output.txt').open('w', encoding='utf-8') as output:
                contender = subprocess.run([*command, 'once'], stdout=output, stderr=subprocess.STDOUT,
                                           env=environment, timeout=15)
            assert contender.returncode == 3
            assert sandbox.client.get('/api/status').get_json()['positions'] == []
        finally:
            owner.kill()  # Deliberate crash: no graceful release/DB state cleanup.
            owner.wait(timeout=15)
            for pipe in (owner.stdin, owner.stdout, owner.stderr):
                if pipe is not None:
                    pipe.close()
        with (tmp_path / 'replacement-output.txt').open('w', encoding='utf-8') as output:
            replacement = subprocess.run([*command, 'once'], stdout=output, stderr=subprocess.STDOUT,
                                         env=environment, timeout=15)
        assert replacement.returncode == 0
        state = sandbox.client.get('/api/status').get_json()
        assert state['positions'][0]['qty'] == '3'
        assert state['ownership']['generation'] == 2


def test_unresolved_prior_dispatch_prevents_new_work_after_handover(tmp_path: Path) -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(tmp_path) as sandbox:
        sandbox.seed_price('BTC/USDT', 100)
        for _ in range(2):
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
        operation = sandbox.client.get('/api/status').get_json()['operations'][0]['operation_id']
        # Disposable crash fixture: an unresolved provider submission is durable.
        with sqlite3.connect(tmp_path / 'trading.sqlite3') as fixture:
            fixture.execute("UPDATE intents SET state='unknown' WHERE operation_id=?", (operation,))
        with ExecutionWorker(sandbox.module.state_store) as replacement:
            assert replacement.process_pending() == 0
            state = sandbox.client.get('/api/status').get_json()
            assert state['positions'] == []
            assert [op['state'] for op in state['operations']] == ['unknown', 'queued']
            assert state['ownership']['state'] == 'recovering'


def test_owned_scan_snapshot_is_shared_across_http_workers(tmp_path: Path) -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(tmp_path) as first:
        with LegacySandbox(tmp_path) as second:
            with ExecutionWorker(first.module.state_store) as owner:
                owner.publish_scan({'last_scan': '2026-10-03T12:00:00+00:00',
                                    'universe': ['BTC/USDT'], 'live_prices': {'BTC/USDT': 100},
                                    'signals': [{'symbol': 'BTC/USDT', 'type': 'hammer', 'time': 'fixture'}]})
            state = second.client.get('/api/status').get_json()
            assert state['universe'] == ['BTC/USDT']
            assert state['live_prices'] == {'BTC/USDT': 100}
            assert state['signals'][0]['type'] == 'hammer'


def test_corrupt_state_and_unconfirmed_or_wrong_namespace_import_fail_closed(tmp_path: Path) -> None:
    import json
    import pytest
    from state_admin import quarantine_legacy, confirm_paper_snapshot
    from state_store import StateStore, StateUnavailable
    source = tmp_path / 'positions.json'
    source.write_text(json.dumps([
        {'symbol': 'BTC/USDT', 'qty': '1', 'entry': '100'},
        {'symbol': 'BTC/USDT', 'qty': '2', 'entry': '100'},
        {'symbol': 'ETH/USDT', 'qty': 'NaN', 'entry': '20'},
    ]), encoding='utf-8')
    with LegacySandbox(tmp_path) as sandbox:
        checksum = quarantine_legacy(sandbox.module.state_store, [source])[0]['checksum']
        with pytest.raises(StateUnavailable):
            confirm_paper_snapshot(sandbox.module.state_store, checksum, actor='operator', origin_confirmation='paper')
        exchange = StateStore(tmp_path, 'operator-account', 'testnet', 'exchange')
        exchange.provision(sandbox.module.DEFAULT_CONFIG)
        with pytest.raises(StateUnavailable):
            confirm_paper_snapshot(exchange, checksum, actor='operator', origin_confirmation=f'paper:{exchange.namespace}:{checksum}')
        confirm_paper_snapshot(sandbox.module.state_store, checksum, actor='operator',
                               origin_confirmation=f'paper:{sandbox.module.state_store.namespace}:{checksum}')
        state = sandbox.client.get('/api/status').get_json()
        assert state['positions'] == []
        assert state['migration'][0]['quarantine_count'] == 3
        database = tmp_path / 'trading.sqlite3'
        database.write_bytes(b'corrupted database')
        assert sandbox.client.get('/api/status').status_code == 503
        assert sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'}).status_code == 503
        with pytest.raises(StateUnavailable):
            sandbox.run_owner()


def test_corrupt_backup_is_rejected_and_legacy_writers_cannot_change_snapshots(tmp_path: Path) -> None:
    import pytest
    from state_admin import backup_state, restore_state, quarantine_legacy
    from state_store import StateUnavailable
    source = tmp_path / 'positions.json'
    source.write_text('[]', encoding='utf-8')
    with LegacySandbox(tmp_path) as sandbox:
        quarantine_legacy(sandbox.module.state_store, [source])
        sandbox.seed_price('BTC/USDT', 100)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'})
        assert source.read_text(encoding='utf-8') == '[]'
        assert not hasattr(sandbox.module, 'save_positions')
        backup = tmp_path / 'backup'
        backup_state(sandbox.module.state_store, backup)
        (backup / 'trading.sqlite3').write_bytes(b'corrupt backup')
        with pytest.raises(StateUnavailable, match='checksum'):
            restore_state(sandbox.module.state_store, backup, tmp_path / 'restored')
        assert sandbox.client.get('/api/status').get_json()['positions'][0]['qty'] == '1'


def test_separate_http_process_reads_committed_positions_and_updates_shared_config(tmp_path: Path) -> None:
    import json
    import os
    import subprocess
    import sys
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('BINANCE_', 'OPERATOR_', 'ALLOW_', 'TRADING_'))
                   and key not in {'API_KEY', 'API_SECRET', 'FLASK_SECRET_KEY', 'DATA_DIR'}}
    with LegacySandbox(tmp_path) as first:
        environment['BT03_OFFLINE_HASH'] = first.module.app.config['OPERATOR_PASSWORD_HASH']
        first.seed_price('BTC/USDT', 100)
        first.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '0.125'})
        output_path = tmp_path / 'http-output.json'
        with output_path.open('w', encoding='utf-8') as output:
            response = subprocess.run([sys.executable, '-m', 'tests.http_process', str(tmp_path), '0.025'],
                                      stdout=output, stderr=subprocess.STDOUT, env=environment, timeout=15)
        assert response.returncode == 0, output_path.read_text(encoding='utf-8')
        state = json.loads(output_path.read_text(encoding='utf-8'))
        assert state['positions'][0]['qty'] == '0.125'
        assert first.client.get('/api/status').get_json()['config']['risk'] == 0.025


def test_changed_owner_fence_prevents_effects_and_preserves_queue(tmp_path: Path) -> None:
    import pytest
    from execution_worker import ExecutionWorker
    from execution_owner import OwnershipUnavailable
    with LegacySandbox(tmp_path) as sandbox:
        sandbox.seed_price('BTC/USDT', 100)
        sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
        with ExecutionWorker(sandbox.module.state_store) as worker:
            with sqlite3.connect(tmp_path / 'trading.sqlite3') as fault:
                fault.execute("UPDATE execution_owners SET token='uncertain-owner',generation=generation+1")
            with pytest.raises(OwnershipUnavailable):
                worker.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['positions'] == []
        assert state['operations'][0]['state'] == 'queued'


def test_backup_metadata_cannot_conceal_missing_newer_journal_effects(tmp_path: Path) -> None:
    import json
    import pytest
    from state_admin import backup_state, restore_state
    from state_store import StateUnavailable
    with LegacySandbox(tmp_path) as sandbox:
        sandbox.seed_price('BTC/USDT', 100)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '2'})
        backup = tmp_path / 'backup'
        backup_state(sandbox.module.state_store, backup)
        sandbox.post('/close', data={'symbol': 'BTC/USDT', 'qty': '1'})
        metadata = backup / 'backup-manifest.json'
        manifest = json.loads(metadata.read_text(encoding='utf-8'))
        manifest['journal_sequence'] = 1000000
        metadata.write_text(json.dumps(manifest), encoding='utf-8')
        with pytest.raises(StateUnavailable, match='metadata'):
            restore_state(sandbox.module.state_store, backup, tmp_path / 'stale-candidate')
        assert sandbox.client.get('/api/status').get_json()['positions'][0]['qty'] == '1'
        assert not (tmp_path / 'stale-candidate').exists()


def test_status_uses_one_snapshot_when_fill_commits_between_database_reads(tmp_path: Path) -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(tmp_path) as sandbox:
        sandbox.seed_price('BTC/USDT', 100)
        sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
        # Public DB-API fault boundary: release the first read, commit an owner
        # effect, then continue HTTP observation. Assertions stay at HTTP.
        injected = False
        with ExecutionWorker(sandbox.module.state_store) as owner:
            class InterleavingRead(sqlite3.Connection):
                read_positions = False
                def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
                    if 'SELECT symbol,qty,entry,basis_status FROM positions' in sql:
                        self.read_positions = True
                    return super().execute(sql, parameters)
                def close(self) -> None:
                    nonlocal injected
                    super().close()
                    if self.read_positions and not injected:
                        injected = True
                        with patch('sqlite3.connect', connect):
                            owner.process_pending()
            connect = sqlite3.connect
            def interleaving_connect(database: str, *, timeout: float = 5,
                                     isolation_level: Literal['DEFERRED', 'EXCLUSIVE', 'IMMEDIATE'] | None = None,
                                     uri: bool = False) -> sqlite3.Connection:
                return connect(database, timeout=timeout, isolation_level=isolation_level, uri=uri, factory=InterleavingRead)
            with patch('sqlite3.connect', interleaving_connect):
                state = sandbox.client.get('/api/status').get_json()
            assert injected
            assert state['open_positions'] == []
            assert state['positions'] == []
            assert state['operations'][0]['state'] == 'queued'
        assert sandbox.client.get('/api/status').get_json()['positions'][0]['qty'] == '1'
