"""BT07 protection observations through the existing owned HTTP seam."""
from decimal import Decimal
from pathlib import Path
from tests.legacy_sandbox import LegacySandbox
from tests.test_paper_execution import setup
from tests.test_exchange_execution import SpotFixture, configure
from typing import Any
import pytest
import time


def enable_protection(sandbox: LegacySandbox) -> None:
    original = sandbox.exchange.metadata
    def metadata() -> dict[str, Any]:
        values = original()
        for symbol in values['symbols']:
            symbol.update(ocoAllowed=True, orderTypes=['MARKET', 'STOP_LOSS', 'TAKE_PROFIT'])
            symbol['filters'].append({'filterType': 'PRICE_FILTER', 'minPrice': '0.01', 'maxPrice': '1000000', 'tickSize': '0.01'})
        return values
    sandbox.exchange.metadata = metadata  # type: ignore[method-assign]


def test_exchange_partial_close_cancels_pair_and_reprotects_confirmed_remainder() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        enable_protection(sandbox)
        sandbox.seed_price('BTC/USDT', 100)
        provider = SpotFixture()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            state = sandbox.client.get('/api/status').get_json()
            assert state['protection'][0]['state'] == 'active'
            assert state['protection'][0]['semantics'] == 'exchange_confirmed'
            sandbox.client.post('/close', data={'symbol': 'BTC/USDT', 'qty': '0.4', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            state = sandbox.client.get('/api/status').get_json()
            assert state['positions'][0]['qty'] == '0.6'
            assert state['protection'][-1]['state'] == 'active'
            assert state['protection'][-1]['quantity'] == '0.6'
        assert len(provider.submissions) == 2
        assert len(provider.protect_submissions) == 2
        assert len(provider.cancel_submissions) == 1


def test_paper_fill_installs_simulated_pair_and_partial_close_preserves_rules(tmp_path: Path) -> None:
    with LegacySandbox(tmp_path, paper_setup=False) as sandbox:
        assert setup(sandbox, fee_fraction='0', slippage_fraction='0') == 302
        sandbox.seed_price('BTC/USDT', 100)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'})
        state = sandbox.client.get('/api/status').get_json()
        group = state['protection'][0]
        assert group['state'] == 'active'
        assert group['semantics'] == 'simulation'
        assert group['quantity'] == '1'
        assert group['stop_price'] == '97'
        assert group['take_profit_price'] == '106'
        sandbox.post('/update-config', data={'stop_loss_pct': '0.1', 'take_profit_pct': '0.2'})
        sandbox.post('/close', data={'symbol': 'BTC/USDT', 'qty': '0.4'})
        state = sandbox.client.get('/api/status').get_json()
        active = [g for g in state['protection'] if g['state'] == 'active']
        assert len(active) == 1
        assert active[0]['quantity'] == '0.6'
        assert active[0]['stop_price'] == '97'
        assert active[0]['take_profit_price'] == '106'
        assert sandbox.exchange.orders == []
    with LegacySandbox(tmp_path, paper_setup=False) as restarted:
        assert restarted.client.get('/api/status').get_json()['protection'][-1]['quantity'] == '0.6'


def test_paper_stop_gap_uses_adverse_price_and_keeps_entry_history() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox, fee_fraction='0.01', slippage_fraction='0.01') == 302
        sandbox.seed_price('BTC/USDT', 100)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'})
        sandbox.seed_price('BTC/USDT', 90)
        with ExecutionWorker(sandbox.module.state_store) as owner:
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['positions'] == []
        assert len(state['fills']) == 2
        assert Decimal(state['fills'][1]['price']) == Decimal('89.1')
        assert state['protection'][-1]['state'] == 'closed'
        assert state['protection'][-1]['exit_reason'] == 'stop_loss'
        assert sandbox.exchange.orders == []


def test_unsupported_spot_protection_refuses_entry_before_private_submission() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        original = sandbox.exchange.metadata
        def unsupported() -> dict[str, Any]:
            metadata = original()
            for market in metadata['symbols']:
                market['ocoAllowed'] = False
            return metadata
        sandbox.exchange.metadata = unsupported  # type: ignore[method-assign]
        sandbox.seed_price('BTC/USDT', 100)
        provider = SpotFixture()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['operations'][0]['error_class'] == 'spot_market_protection_unsupported'
        assert provider.submissions == []
        assert state['fills'] == []


def test_protection_unknown_retains_entry_blocks_buys_and_recovers_by_original_list() -> None:
    from execution_worker import ExecutionWorker
    class LostList(SpotFixture):
        visible = False
        def query_list(self, list_id: str) -> dict[str, Any] | None:
            return super().query_list(list_id) if self.visible else None
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox, cooldown_seconds='0')
        sandbox.seed_price('BTC/USDT', 100)
        provider = LostList()
        provider.timeout_install = True
        tick = [float(int(time.time()))]
        with ExecutionWorker(sandbox.module.state_store, clock=lambda: tick[0], exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            state = sandbox.client.get('/api/status').get_json()
            assert state['operations'][0]['state'] == 'filled'
            assert state['protection'][-1]['state'] == 'unknown'
            assert state['exchange_risk']['entry_blocked'] is True
            assert state['positions'][0]['qty'] == '1'
            tick[0] += 15
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            assert len(provider.submissions) == 1
            assert 'Reconcile this protection list' in sandbox.client.get('/').get_data(as_text=True)
            provider.visible = True
            owner.reconcile_exchange()
            state = sandbox.client.get('/api/status').get_json()
            assert state['protection'][-1]['state'] == 'active'
            assert len(provider.protect_submissions) == 1


def test_rejected_protection_preserves_entry_and_permits_reviewed_full_reduction() -> None:
    from exchange_adapter import SubmissionRejected
    from execution_worker import ExecutionWorker
    class RejectedList(SpotFixture):
        def protect(self, *args: Any) -> dict[str, Any]:
            raise SubmissionRejected('invalid_market')
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100)
        provider = RejectedList()
        tick = [float(int(time.time()))]
        with ExecutionWorker(sandbox.module.state_store, clock=lambda: tick[0], exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            state = sandbox.client.get('/api/status').get_json()
            assert state['protection'][-1]['state'] == 'failed'
            assert state['positions'][0]['qty'] == '1'
            assert state['operations'][0]['state'] == 'filled'
            assert state['exchange_risk']['entry_blocked'] is True
            tick[0] += 15
            sandbox.client.post('/close', data={'symbol': 'BTC/USDT', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            state = sandbox.client.get('/api/status').get_json()
            assert state['positions'] == []
            assert len(state['fills']) == 2
            assert state['protection'][-1]['state'] == 'closed'


def test_cancel_timeout_restart_and_duplicate_close_never_issue_conflicting_sell(tmp_path: Path) -> None:
    from execution_worker import ExecutionWorker
    environment = {'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}
    provider = SpotFixture()
    provider.timeout_cancel = True
    with LegacySandbox(tmp_path, paper_setup=False, environment=environment) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100)
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            for _ in range(2):
                sandbox.client.post('/close', data={'symbol': 'BTC/USDT', 'idempotency_key': 'same-close', 'csrf_token': sandbox.csrf})
                owner.process_pending()
    with LegacySandbox(tmp_path, paper_setup=False, environment=environment) as restarted:
        restarted.seed_price('BTC/USDT', 100)
        with ExecutionWorker(restarted.module.state_store, exchange_provider=provider) as owner:
            owner.process_pending()
        state = restarted.client.get('/api/status').get_json()
        assert state['protection'][-1]['state'] == 'canceling'
        assert state['operations'][-1]['state'] == 'queued'
        assert state['exchange_risk']['entry_blocked'] is True
        assert len(provider.cancel_submissions) == 1
        assert len(provider.submissions) == 1
        assert len(state['operations']) == 2


def fill_stop(provider: SpotFixture, group: dict[str, Any], quantity: str) -> None:
    order = provider.orders[group['orders'][1]['clientOrderId']]
    qty, price = Decimal(quantity), Decimal('90')
    fee = qty*price*Decimal('0.002')
    provider.balances['BTC'] -= qty
    provider.balances['USDT'] += qty*price-fee
    order.update(status='FILLED' if quantity == group['quantity'] else 'PARTIALLY_FILLED',
                 executedQty=quantity, cummulativeQuoteQty=str(qty*price), updateTime=int(time.time()*1000))
    provider.fills.append({'symbol': 'BTCUSDT', 'id': len(provider.fills)+1, 'orderId': order['orderId'],
                           'price': str(price), 'qty': quantity, 'quoteQty': str(qty*price), 'commission': str(fee),
                           'commissionAsset': 'USDT', 'isBuyer': False, 'time': order['updateTime']})
    provider.orders[group['orders'][0]['clientOrderId']]['status'] = 'CANCELED'
    if quantity == group['quantity']:
        group['listOrderStatus'] = 'ALL_DONE'


@pytest.mark.parametrize(('fill_quantity', 'full_close'), [('0.3', False), ('1', False), ('0.3', True), ('1', True)])
def test_cancel_fill_race_never_sells_beyond_confirmed_remainder(fill_quantity: str, full_close: bool) -> None:
    from execution_worker import ExecutionWorker
    class RacingList(SpotFixture):
        def cancel_list(self, symbol: str, list_id: str, cancel_id: str) -> dict[str, Any]:
            fill_stop(self, self.lists[list_id], fill_quantity)
            return super().cancel_list(symbol, list_id, cancel_id)
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100)
        provider = RacingList()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            close = {'symbol': 'BTC/USDT', 'csrf_token': sandbox.csrf}
            if not full_close:
                close['qty'] = '0.4'
            sandbox.client.post('/close', data=close)
            owner.process_pending()
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert sum(Decimal(f['qty']) for f in state['fills'] if f['side'] == 'sell') <= 1
        assert len(provider.submissions) == (2 if fill_quantity == '0.3' else 1)
        if fill_quantity == '0.3' and not full_close:
            assert state['positions'][0]['qty'] == '0.3'
            assert state['protection'][-1]['quantity'] == '0.3'
            assert state['protection'][-1]['state'] == 'active'
        elif fill_quantity == '1':
            assert state['positions'] == []
            assert state['operations'][-1]['state'] == 'rejected'
        else:
            assert state['positions'] == []
            assert state['operations'][-1]['state'] == 'filled'
            assert state['operations'][-1]['requested_qty'] == '1'
            assert state['fills'][-1]['qty'] == '0.7'


def test_canceled_provider_client_alias_retains_verified_leg_identity() -> None:
    from execution_worker import ExecutionWorker
    class AliasedCancel(SpotFixture):
        def cancel_list(self, symbol: str, list_id: str, cancel_id: str) -> dict[str, Any]:
            group = super().cancel_list(symbol, list_id, cancel_id)
            for leg in group['orders']:
                self.orders[leg['clientOrderId']]['clientOrderId'] = cancel_id
            return group
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100)
        provider = AliasedCancel()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            sandbox.client.post('/close', data={'symbol': 'BTC/USDT', 'csrf_token': sandbox.csrf})
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['positions'] == []
        assert state['protection'][-1]['leg_states'] == ['canceled', 'canceled']
        assert len(provider.submissions) == 2


def test_completed_paper_bar_with_both_targets_touched_uses_stop_first() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    from tests.test_market_runtime import ControlledClock, MarketFixture
    class AmbiguousBars(MarketFixture):
        def candles(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
            rows = super().candles(symbol, timeframe, limit)
            if timeframe == '15m':
                rows[-1][2:4] = [110, 90]
            return rows
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox, fee_fraction='0', slippage_fraction='0') == 302
        clock = ControlledClock()
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            MarketRuntime(owner, MarketFixture(clock()), clock=clock).scan_once()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            clock.value += 900
            MarketRuntime(owner, AmbiguousBars(clock()), clock=clock).scan_once()
            owner.process_pending()
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['positions'] == []
        assert len(state['fills']) == 2
        assert state['fills'][-1]['price'] == '97'
        assert state['protection'][-1]['exit_reason'] == 'stop_loss'


def test_duplicate_distinct_full_close_requests_do_not_double_sell() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100)
        provider = SpotFixture()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            for key in ('close-a', 'close-b'):
                sandbox.client.post('/close', data={'symbol': 'BTC/USDT', 'idempotency_key': key, 'csrf_token': sandbox.csrf})
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['positions'] == []
        assert [o['state'] for o in state['operations']] == ['filled', 'filled', 'rejected']
        assert len(provider.submissions) == 2


def test_entry_pause_and_stopped_owner_preserve_confirmed_provider_pair(tmp_path: Path) -> None:
    from execution_worker import ExecutionWorker
    environment = {'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}
    provider = SpotFixture()
    with LegacySandbox(tmp_path, paper_setup=False, environment=environment) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100)
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            sandbox.client.post('/exchange/entries', data={'action': 'pause', 'csrf_token': sandbox.csrf})
        state = sandbox.client.get('/api/status').get_json()
        assert state['ownership']['state'] == 'stopped'
        assert state['protection'][-1]['state'] == 'active'
        assert provider.cancel_submissions == []
    with LegacySandbox(tmp_path, paper_setup=False, environment=environment) as restarted:
        restarted.seed_price('BTC/USDT', 100)
        with ExecutionWorker(restarted.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            restarted.client.post('/close', data={'symbol': 'BTC/USDT', 'qty': '0.5', 'csrf_token': restarted.csrf})
            owner.process_pending()
        state = restarted.client.get('/api/status').get_json()
        assert state['entry_control']['paused'] is True
        assert state['positions'][0]['qty'] == '0.5'
        assert state['protection'][-1]['state'] == 'active'
        assert len(provider.protect_submissions) == 2


def test_partial_entry_fill_is_protected_while_unfilled_buy_stays_reserved() -> None:
    from execution_worker import ExecutionWorker
    class PartialEntry(SpotFixture):
        def submit(self, symbol: str, side: str, quantity: str, client_id: str) -> dict[str, Any]:
            order = super().submit(symbol, side, '0.4', client_id)
            order.update(origQty=quantity, status='PARTIALLY_FILLED')
            self.orders[client_id] = order
            return order
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100)
        provider = PartialEntry()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['operations'][0]['state'] == 'partially_filled'
        assert state['positions'][0]['qty'] == '0.4'
        assert state['protection'][-1]['quantity'] == '0.4'
        assert state['protection'][-1]['state'] == 'active'
        assert state['exchange_account']['balances']['USDT']['reserved'] == '61.206'
        assert state['exchange_risk']['entry_blocked'] is True


def test_budget_deferred_protection_has_no_dispatch_and_allows_reduction_after_recovery() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    from tests.test_market_runtime import MarketFixture
    class ExhaustedPlacement(SpotFixture):
        account_receipts = 0
        exhaust = True
        def account(self) -> dict[str, Any]:
            self.headers = {}
            if self.submissions and not self.lists:
                self.account_receipts += 1
                if self.exhaust and self.account_receipts == 2:
                    self.headers = {'x-mbx-used-weight-1m': '100000'}
            return super().account()
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100)
        provider = ExhaustedPlacement()
        tick = [float(int(time.time()))]
        with ExecutionWorker(sandbox.module.state_store, clock=lambda: tick[0], exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            state = sandbox.client.get('/api/status').get_json()
            assert provider.protect_submissions == []
            assert state['operations'][0]['state'] == 'filled'
            assert state['protection'][-1]['state'] == 'failed'
            assert state['protection'][-1]['dispatch_attempted'] is False
            assert state['exchange_account']['balances']['BTC']['reserved'] == '0'
            provider.exhaust = False
            provider.headers = {}
            tick[0] += 61
            MarketRuntime(owner, MarketFixture(tick[0]), clock=lambda: tick[0]).scan_once()
            sandbox.client.post('/close', data={'symbol': 'BTC/USDT', 'csrf_token': sandbox.csrf})
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['positions'] == []
        assert state['operations'][-1]['state'] == 'filled'
        assert len(provider.submissions) == 2


def test_budget_deferred_cancel_retains_active_pair_and_retries_after_recovery() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    from tests.test_market_runtime import MarketFixture
    class ExhaustedCancel(SpotFixture):
        exhaust = False
        trade_receipts = 0
        def trades(self, symbol: str, *, from_id: int = 0, order_id: str | None = None) -> list[dict[str, Any]]:
            self.headers = {}
            if self.exhaust and order_id is not None:
                self.trade_receipts += 1
                # Reconciliation first verifies both legs and observes both open
                # orders plus buy history (five); close preparation adds two.
                if self.trade_receipts == 7:
                    self.headers = {'x-mbx-used-weight-1m': '100000'}
            return super().trades(symbol, from_id=from_id, order_id=order_id)
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100)
        provider = ExhaustedCancel()
        tick = [float(int(time.time()))]
        if tick[0] % 60 == 0:
            tick[0] += 1
        with ExecutionWorker(sandbox.module.state_store, clock=lambda: tick[0], exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            provider.exhaust = True
            sandbox.client.post('/close', data={'symbol': 'BTC/USDT', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            state = sandbox.client.get('/api/status').get_json()
            assert provider.cancel_submissions == []
            assert state['protection'][-1]['state'] == 'active'
            assert state['operations'][-1]['state'] == 'queued'
            provider.exhaust = False
            provider.headers = {}
            tick[0] += 60-int(tick[0] % 60)
            MarketRuntime(owner, MarketFixture(tick[0]), clock=lambda: tick[0]).scan_once()
            owner.process_pending()
        assert sandbox.client.get('/api/status').get_json()['positions'] == []
        assert len(provider.cancel_submissions) == 1
        assert len(provider.submissions) == 2


def test_crash_before_protection_budget_admission_recovers_proven_no_dispatch() -> None:
    from execution_worker import ExecutionWorker
    class ProcessCrash(BaseException):
        pass
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100)
        provider = SpotFixture()
        with pytest.raises(ProcessCrash):
            with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
                owner.reconcile_exchange()
                execution = owner.exchange_execution
                assert execution is not None
                request = execution._request
                def interrupted(lane: str, *args: Any, **kwargs: Any) -> Any:
                    if lane == 'reduction':
                        raise ProcessCrash()
                    return request(lane, *args, **kwargs)
                execution._request = interrupted  # type: ignore[method-assign]
                sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
                owner.process_pending()
        assert provider.protect_submissions == []
        state = sandbox.client.get('/api/status').get_json()
        assert state['protection'][-1]['state'] == 'installing'
        assert state['protection'][-1]['dispatch_attempted'] is False
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as recovered:
            recovered.reconcile_exchange()
            state = sandbox.client.get('/api/status').get_json()
            assert state['protection'][-1]['state'] == 'failed'
            assert state['exchange_account']['balances']['BTC']['reserved'] == '0'
            sandbox.client.post('/close', data={'symbol': 'BTC/USDT', 'csrf_token': sandbox.csrf})
            recovered.process_pending()
        assert sandbox.client.get('/api/status').get_json()['positions'] == []
        assert len(provider.submissions) == 2
