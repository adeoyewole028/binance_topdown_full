"""BT05 acceptance through authenticated HTTP, owned worker and public fixtures."""
from tests.legacy_sandbox import LegacySandbox
import pytest
from typing import Any
from pathlib import Path


def test_unconfigured_paper_account_cannot_fill_or_fabricate_cash() -> None:
    with LegacySandbox(paper_setup=False) as sandbox:
        sandbox.seed_price('BTC/USDT', 100.0)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'})
        state = sandbox.client.get('/api/status').get_json()
        assert state['paper_account']['state'] == 'setup_required'
        assert state['paper_account']['balances'] == {}
        assert state['positions'] == []
        assert state['operations'][0]['state'] == 'waiting_for_setup'


PAPER_INPUTS = {
    'quote': 'USDT', 'initial_cash': '1000', 'fee_fraction': '0.01', 'slippage_fraction': '0.01',
    'risk_fraction': '0.1', 'fixed_quote': '100', 'max_allocation_fraction': '0.5',
    'max_exposure_fraction': '0.8', 'max_positions': '2', 'cooldown_seconds': '120',
    'drawdown_fraction': '0.2',
}


def setup(sandbox: LegacySandbox, **changes: str) -> int:
    version = sandbox.client.get('/api/status').get_json()['config_version']
    return int(sandbox.post('/paper/setup', data={**PAPER_INPUTS, **changes, 'expected_config_version': str(version)}).status_code)


def test_explicit_paper_setup_creates_one_seed_and_cannot_refund_on_repeat() -> None:
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox) == 302
        account = sandbox.client.get('/api/status').get_json()['paper_account']
        assert account['balances'] == {'USDT': {'total': '1000', 'reserved': '0', 'available': '1000'}}
        assert setup(sandbox, initial_cash='2000') == 409
        assert sandbox.client.get('/api/status').get_json()['paper_account']['balances'] == account['balances']


def test_paper_buy_accounts_for_adverse_fill_and_quote_fee_atomically() -> None:
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox) == 302
        sandbox.seed_price('BTC/USDT', 100.0)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'})
        state = sandbox.client.get('/api/status').get_json()
        assert state['paper_account']['balances'] == {
            'USDT': {'total': '897.99', 'reserved': '0', 'available': '897.99'},
            'BTC': {'total': '1', 'reserved': '0', 'available': '1'},
        }
        assert state['positions'][0]['qty'] == '1'
        assert state['positions'][0]['entry'] == '102.01'
        assert state['operations'][0]['state'] == 'filled'


def test_repeated_paper_request_has_one_effect_and_semantic_conflict_is_visible() -> None:
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox) == 302
        sandbox.seed_price('BTC/USDT', 100.0)
        order = {'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'idempotency_key': 'same-paper-order'}
        sandbox.post('/trade', data=order)
        sandbox.post('/trade', data={**order, 'qty': '1.00'})
        assert sandbox.post('/trade', data={**order, 'qty': '2'}).status_code == 409
        state = sandbox.client.get('/api/status').get_json()
        assert len(state['operations']) == 1
        assert state['positions'][0]['qty'] == '1'
        assert state['paper_account']['balances']['USDT']['total'] == '897.99'


def test_pending_buys_reserve_cash_and_reject_combined_overspend() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox, max_allocation_fraction='1', max_exposure_fraction='1') == 302
        sandbox.seed_price('BTC/USDT', 100.0)
        with ExecutionWorker(sandbox.module.state_store) as owner:
            for key in ('first', 'second'):
                sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '6',
                                                   'idempotency_key': key, 'csrf_token': sandbox.csrf})
            state = sandbox.client.get('/api/status').get_json()
            assert state['paper_account']['balances']['USDT']['reserved'] == '612.06'
            assert state['operations'][1]['state'] == 'rejected'
            owner.process_pending()
            state = sandbox.client.get('/api/status').get_json()
            assert state['paper_account']['balances']['USDT']['total'] == '387.94'
            assert state['paper_account']['balances']['USDT']['reserved'] == '0'
            assert state['positions'][0]['qty'] == '6'


@pytest.mark.parametrize('quantity', ['0.0005', '0.01'])
def test_exchange_quantity_and_notional_rules_reject_without_effects(quantity: str) -> None:
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox) == 302
        sandbox.seed_price('BTC/USDT', 100.0)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': quantity})
        state = sandbox.client.get('/api/status').get_json()
        assert state['positions'] == []
        assert state['paper_account']['balances']['USDT']['total'] == '1000'
        assert state['operations'][0]['state'] == 'rejected'


def test_paper_market_notional_uses_verified_average_window() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    from tests.legacy_sandbox import FakeExchange
    class AverageProvider(FakeExchange):
        def metadata(self) -> dict[str, Any]:
            result = super().metadata()
            result['symbols'][0]['filters'][1]['avgPriceMins'] = 5
            return result
        def reference_price(self, symbol: str) -> dict[str, Any]:
            return {'referencePrice': None}
        def average_price(self, symbol: str) -> dict[str, Any]:
            return {'mins': 5, 'price': '50'}
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox) == 302
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = AverageProvider()
        provider.prices = {'BTC/USDT': 100.0}
        with ExecutionWorker(sandbox.module.state_store) as owner:
            MarketRuntime(owner, provider).scan_once()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            assert sandbox.client.get('/api/status').get_json()['operations'][0]['state'] == 'filled'


@pytest.mark.parametrize(('changes', 'quantity'), [
    ({'max_allocation_fraction': '0.05'}, '1'),
    ({'max_exposure_fraction': '0.05'}, '1'),
    ({'risk_fraction': '0.001'}, '1'),
])
def test_explicit_entry_caps_reject_with_cash_and_holdings_unchanged(changes: dict[str, str], quantity: str) -> None:
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox, **changes) == 302
        sandbox.seed_price('BTC/USDT', 100.0)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': quantity})
        state = sandbox.client.get('/api/status').get_json()
        assert state['operations'][0]['state'] == 'rejected'
        assert state['positions'] == []
        assert state['paper_account']['balances']['USDT']['available'] == '1000'


def test_entry_cooldown_survives_restart_and_does_not_block_partial_risk_reduction(tmp_path: Path) -> None:
    with LegacySandbox(tmp_path, paper_setup=False) as sandbox:
        assert setup(sandbox) == 302
        sandbox.seed_price('BTC/USDT', 100.0)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'})
    with LegacySandbox(tmp_path, paper_setup=False) as restarted:
        restarted.seed_price('BTC/USDT', 100.0)
        restarted.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'})
        assert restarted.client.get('/api/status').get_json()['operations'][1]['state'] == 'rejected'
        restarted.post('/close', data={'symbol': 'BTC/USDT', 'qty': '0.5'})
        state = restarted.client.get('/api/status').get_json()
        assert state['positions'][0]['qty'] == '0.5'
        assert state['positions'][0]['entry'] == '102.01'
        assert state['paper_account']['balances']['USDT']['total'] == '946.995'
        assert state['operations'][2]['state'] == 'filled'


def test_pending_entries_reserve_distinct_position_slots() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox, max_positions='1') == 302
        sandbox.seed_price('BTC/USDT', 100.0)
        sandbox.seed_price('ETH/USDT', 100.0)
        with ExecutionWorker(sandbox.module.state_store) as owner:
            for symbol in ('BTC/USDT', 'ETH/USDT'):
                sandbox.client.post('/trade', data={'symbol': symbol, 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            state = sandbox.client.get('/api/status').get_json()
            assert [operation['state'] for operation in state['operations']] == ['queued', 'rejected']
            owner.process_pending()
            assert sandbox.client.get('/api/status').get_json()['positions'][0]['symbol'] == 'BTC/USDT'


def test_fixed_sizing_is_recomputed_from_funded_quote_instead_of_ui_quantity() -> None:
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox) == 302
        sandbox.seed_price('BTC/USDT', 100.0)
        sandbox.post('/update-config', data={'equity_proxy': '99999999'})
        sandbox.run_owner()
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '999999', 'sizing': 'fixed'})
        state = sandbox.client.get('/api/status').get_json()
        assert state['operations'][0]['state'] == 'filled'
        assert state['positions'][0]['qty'] == '0.98'
        assert state['paper_account']['balances']['USDT']['total'] == '900.0302'


def test_changed_price_requires_review_and_releases_the_old_reservation() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox) == 302
        sandbox.seed_price('BTC/USDT', 100.0)
        with ExecutionWorker(sandbox.module.state_store) as owner:
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            sandbox.exchange.prices['BTC/USDT'] = 110.0
            MarketRuntime(owner, sandbox.exchange).refresh_prices()
            owner.process_pending()
            state = sandbox.client.get('/api/status').get_json()
            assert state['operations'][0]['state'] == 'rejected'
            assert state['positions'] == []
            assert state['paper_account']['balances']['USDT']['reserved'] == '0'


def test_drawdown_stop_blocks_entries_but_preserves_manual_risk_reduction() -> None:
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox, fee_fraction='0', slippage_fraction='0', risk_fraction='1',
                     max_allocation_fraction='1', max_exposure_fraction='1', cooldown_seconds='0') == 302
        sandbox.seed_price('BTC/USDT', 100.0)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '6'})
        sandbox.seed_price('BTC/USDT', 20.0)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'})
        state = sandbox.client.get('/api/status').get_json()
        assert state['operations'][1]['state'] == 'rejected'
        assert state['paper_account']['entry_blocked'] is True
        assert state['positions'][0]['qty'] == '6'
        sandbox.post('/close', data={'symbol': 'BTC/USDT', 'qty': '1'})
        state = sandbox.client.get('/api/status').get_json()
        assert state['positions'][0]['qty'] == '5'
        assert state['paper_account']['balances']['USDT']['total'] == '420'


def test_independent_quote_refresh_also_refreshes_required_notional_inputs() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    from tests.test_market_runtime import ControlledClock, MarketFixture
    clock = ControlledClock()
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox) == 302
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            scanner = MarketRuntime(owner, MarketFixture(clock()), clock=clock)
            scanner.scan_once()
            clock.value += 61
            scanner.refresh_prices()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            assert sandbox.client.get('/api/status').get_json()['operations'][0]['state'] == 'filled'


def test_operator_can_review_setup_and_rejected_orders_without_a_success_claim() -> None:
    with LegacySandbox(paper_setup=False) as sandbox:
        page = sandbox.client.get('/').get_data(as_text=True)
        assert 'name="fee_fraction"' in page
        assert 'Paper setup required' in page
        assert setup(sandbox) == 302
        sandbox.seed_price('BTC/USDT', 100.0)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '100'})
        page = sandbox.client.get('/').get_data(as_text=True)
        assert 'insufficient_paper_assets' in page
        assert 'Order rejected' in page
        assert 'Order queued for the execution owner' not in page


def test_risk_sizing_uses_funded_equity_and_rounds_down_to_valid_quantity() -> None:
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox, fee_fraction='0', slippage_fraction='0', risk_fraction='0.01') == 302
        sandbox.seed_price('BTC/USDT', 100.0)
        sandbox.run_owner()
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '999999', 'sizing': 'risk'})
        state = sandbox.client.get('/api/status').get_json()
        assert state['positions'][0]['qty'] == '3.333'
        assert state['paper_account']['balances']['USDT']['total'] == '666.7'


@pytest.mark.parametrize('quantity', ['0', '-1', 'NaN', 'Infinity', '1e999999'])
def test_invalid_quantity_has_no_acceptance_or_economic_effect(quantity: str) -> None:
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox) == 302
        before = sandbox.client.get('/api/status').get_json()['paper_account']['balances']
        assert sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': quantity}).status_code == 400
        state = sandbox.client.get('/api/status').get_json()
        assert state['operations'] == []
        assert state['positions'] == []
        assert state['paper_account']['balances'] == before


def test_oversell_and_duplicate_close_cannot_spend_the_same_holdings() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox) == 302
        sandbox.seed_price('BTC/USDT', 100.0)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'})
        with ExecutionWorker(sandbox.module.state_store) as owner:
            for key in ('close-one', 'close-two'):
                sandbox.client.post('/close', data={'symbol': 'BTC/USDT', 'qty': '0.6', 'idempotency_key': key, 'csrf_token': sandbox.csrf})
            assert sandbox.client.get('/api/status').get_json()['paper_account']['balances']['BTC']['reserved'] == '0.6'
            owner.process_pending()
            state = sandbox.client.get('/api/status').get_json()
            assert state['operations'][2]['state'] == 'rejected'
            assert state['positions'][0]['qty'] == '0.4'
            assert state['paper_account']['balances']['BTC']['available'] == '0.4'


def test_fees_and_weighted_average_basis_remain_observable_after_partial_sale() -> None:
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox, cooldown_seconds='0') == 302
        sandbox.seed_price('BTC/USDT', 100.0)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'})
        sandbox.seed_price('BTC/USDT', 200.0)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'})
        sandbox.post('/close', data={'symbol': 'BTC/USDT', 'qty': '1'})
        state = sandbox.client.get('/api/status').get_json()
        assert state['positions'][0]['entry'] == '153.015'
        assert state['paper_account']['balances']['USDT']['total'] == '889.99'
        assert state['fills'][-1]['fee_amount'] == '1.98'
        assert state['fills'][-1]['fee_asset'] == 'USDT'


@pytest.mark.parametrize('repeat_key', [False, True])
def test_simultaneous_authenticated_buys_cannot_overdraw_or_duplicate(repeat_key: bool) -> None:
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from unittest.mock import patch
    from execution_worker import ExecutionWorker
    original_start = threading.Thread.start
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox, max_allocation_fraction='1', max_exposure_fraction='1') == 302
        sandbox.seed_price('BTC/USDT', 100.0)
        cookie = sandbox.client.get_cookie('session')
        assert cookie is not None
        clients = [sandbox.module.app.test_client() for _ in range(3)]
        for client in clients:
            client.set_cookie('session', cookie.value)
        barrier = threading.Barrier(3)
        def submit(index: int) -> int:
            barrier.wait(timeout=5)
            response = clients[index].post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '6',
                'idempotency_key': 'same-concurrent' if repeat_key else f'concurrent-{index}', 'csrf_token': sandbox.csrf})
            return int(response.status_code)
        with ExecutionWorker(sandbox.module.state_store) as owner:
            # Explicit test-client concurrency only. Provider/DNS/file guards stay active.
            with patch('threading.Thread.start', original_start), ThreadPoolExecutor(max_workers=3) as pool:
                assert list(pool.map(submit, range(3))) == [302, 302, 302]
            state = sandbox.client.get('/api/status').get_json()
            assert len(state['operations']) == (1 if repeat_key else 3)
            assert sum(o['state'] == 'queued' for o in state['operations']) == 1
            assert state['paper_account']['balances']['USDT']['reserved'] == '612.06'
            owner.process_pending()
            state = sandbox.client.get('/api/status').get_json()
            assert len(state['fills']) == 1
            assert state['paper_account']['balances']['USDT']['available'] == '387.94'
            assert sandbox.network_attempts == 0


def test_repeat_close_all_returns_original_operation_after_holdings_disappear() -> None:
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox) == 302
        sandbox.seed_price('BTC/USDT', 100.0)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'})
        request = {'symbol': 'BTC/USDT', 'idempotency_key': 'repeat-full-close'}
        assert sandbox.post('/close', data=request).status_code == 302
        assert sandbox.post('/close', data=request).status_code == 302
        state = sandbox.client.get('/api/status').get_json()
        assert state['positions'] == []
        assert len(state['operations']) == 2
        assert len(state['fills']) == 2


def test_expired_review_price_cannot_lock_funds_or_later_fill_at_the_same_price() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    from tests.test_market_runtime import ControlledClock, MarketFixture
    clock = ControlledClock()
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox) == 302
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            scanner = MarketRuntime(owner, MarketFixture(clock()), clock=clock)
            scanner.scan_once()
            clock.value += 61
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            state = sandbox.client.get('/api/status').get_json()
            assert state['operations'][0]['state'] == 'rejected'
            assert state['paper_account']['balances']['USDT']['available'] == '1000'
            scanner.refresh_prices()
            owner.process_pending()
            assert sandbox.client.get('/api/status').get_json()['positions'] == []


def test_cancel_pending_paper_order_releases_reservation_without_effect() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox) == 302
        sandbox.seed_price('BTC/USDT', 100.0)
        with ExecutionWorker(sandbox.module.state_store) as owner:
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            operation = sandbox.client.get('/api/status').get_json()['operations'][0]['operation_id']
            assert sandbox.post('/paper/cancel', data={'operation_id': operation}).status_code == 302
            assert sandbox.post('/paper/cancel', data={'operation_id': operation}).status_code == 302
            owner.process_pending()
            state = sandbox.client.get('/api/status').get_json()
            assert state['operations'][0]['state'] == 'canceled'
            assert state['paper_account']['balances']['USDT']['available'] == '1000'
            assert state['fills'] == []


@pytest.mark.parametrize(('field', 'value'), [('initial_cash','0'), ('fee_fraction',''), ('fee_fraction','1'),
    ('slippage_fraction','NaN'), ('drawdown_fraction','0'), ('max_positions','1.5'), ('quote','../USDT')])
def test_invalid_setup_cannot_seed_or_change_configuration(field: str, value: str) -> None:
    with LegacySandbox(paper_setup=False) as sandbox:
        before = sandbox.client.get('/api/status').get_json()
        assert setup(sandbox, **{field: value}) == 400
        after = sandbox.client.get('/api/status').get_json()
        assert after['config_version'] == before['config_version']
        assert after['paper_account']['state'] == 'setup_required'
        assert after['paper_account']['balances'] == {}


def test_setup_requires_authenticated_csrf_verified_operator() -> None:
    with LegacySandbox(authenticated=False, paper_setup=False) as sandbox:
        assert sandbox.client.post('/paper/setup', data={**PAPER_INPUTS, 'expected_config_version':'1'}).status_code == 401
        sandbox.login()
        assert sandbox.client.post('/paper/setup', data={**PAPER_INPUTS, 'expected_config_version':'1'}).status_code == 403
        assert sandbox.client.get('/api/status').get_json()['paper_account']['balances'] == {}


def test_failed_seed_commit_leaves_account_unfunded_and_version_unchanged() -> None:
    import sqlite3
    from unittest.mock import patch
    from typing import Literal
    class FailedCommit(sqlite3.Connection):
        def commit(self) -> None:
            raise sqlite3.OperationalError('offline disk-full failure')
    connect = sqlite3.connect
    def fail_commit(database: str, *, timeout: float = 5,
                    isolation_level: Literal['DEFERRED','EXCLUSIVE','IMMEDIATE'] | None = None,
                    uri: bool = False) -> sqlite3.Connection:
        return connect(database, timeout=timeout, isolation_level=isolation_level, uri=uri, factory=FailedCommit)
    with LegacySandbox(paper_setup=False) as sandbox:
        before = sandbox.client.get('/api/status').get_json()
        with patch('sqlite3.connect', fail_commit):
            assert setup(sandbox) == 503
        after = sandbox.client.get('/api/status').get_json()
        assert after['config_version'] == before['config_version']
        assert after['paper_account']['balances'] == {}
        assert setup(sandbox) == 302
        assert sandbox.client.get('/api/status').get_json()['paper_account']['balances']['USDT']['total'] == '1000'


def test_equal_decimal_seed_spellings_never_create_a_second_seed() -> None:
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox) == 302
        assert setup(sandbox, initial_cash='1000.00') == 302
        assert sandbox.client.get('/api/status').get_json()['paper_account']['balances']['USDT']['total'] == '1000'
