"""BT06 owned Spot lifecycle through authenticated HTTP and fake transport."""
from decimal import Decimal
from pathlib import Path
from typing import Any
import time
import pytest
from tests.legacy_sandbox import LegacySandbox
from tests.test_paper_execution import PAPER_INPUTS


def configure(sandbox: LegacySandbox, **changes: str) -> None:
    version = sandbox.client.get('/api/status').get_json()['config_version']
    policy = {k: v for k, v in PAPER_INPUTS.items() if k != 'initial_cash'}
    assert sandbox.post('/exchange/setup', data={**policy, **changes, 'expected_config_version': str(version)}).status_code == 302


class SpotFixture:
    venue = 'testnet'
    uid = 'offline-account'
    headers: dict[str, Any] = {}

    def __init__(self) -> None:
        self.balances = {'USDT': Decimal('1000'), 'BTC': Decimal('0')}
        self.orders: dict[str, dict[str, Any]] = {}
        self.fills: list[dict[str, Any]] = []
        self.submissions: list[str] = []
        self.lists: dict[str, dict[str, Any]] = {}
        self.protect_submissions: list[str] = []
        self.cancel_submissions: list[str] = []
        self.timeout_install = False
        self.timeout_cancel = False

    def protect(self, symbol: str, quantity: str, stop: str, target: str,
                list_id: str, above_id: str, below_id: str) -> dict[str, Any]:
        self.protect_submissions.append(list_id)
        group: dict[str, Any] = {'symbol': symbol.replace('/', ''), 'orderListId': len(self.lists)+1,
                                 'listClientOrderId': list_id, 'listOrderStatus': 'EXECUTING',
                                 'contingencyType': 'OCO', 'quantity': quantity, 'orders': []}
        for client, kind, trigger in ((above_id, 'TAKE_PROFIT', target), (below_id, 'STOP_LOSS', stop)):
            order = {'symbol': group['symbol'], 'orderId': len(self.orders)+1, 'clientOrderId': client,
                     'orderListId': group['orderListId'], 'side': 'SELL', 'type': kind, 'stopPrice': trigger,
                     'status': 'NEW', 'origQty': quantity, 'executedQty': '0', 'cummulativeQuoteQty': '0',
                     'updateTime': int(time.time()*1000)}
            self.orders[client] = order
            group['orders'].append({'symbol': order['symbol'], 'orderId': order['orderId'], 'clientOrderId': client})
        self.lists[list_id] = group
        if self.timeout_install:
            raise TimeoutError('Offline protection response lost')
        return dict(group)

    def query_list(self, list_id: str) -> dict[str, Any] | None:
        return self.lists.get(list_id)

    def cancel_list(self, symbol: str, list_id: str, cancel_id: str) -> dict[str, Any]:
        self.cancel_submissions.append(list_id)
        if self.timeout_cancel:
            raise TimeoutError('Offline cancellation unresolved')
        group = self.lists[list_id]
        group['listOrderStatus'] = 'ALL_DONE'
        for leg in group['orders']:
            order = self.orders[leg['clientOrderId']]
            if order['status'] in {'NEW', 'PARTIALLY_FILLED'}:
                order['status'] = 'CANCELED'
        return dict(group)


    def configure(self, metadata: dict[str, Any]) -> None:
        pass

    def quantity(self, symbol: str, value: str) -> str:
        return value

    def account(self) -> dict[str, Any]:
        locked = min(self.balances.get('BTC', Decimal(0)), sum((Decimal(group['quantity']) for group in self.lists.values() if group['listOrderStatus'] == 'EXECUTING'), Decimal(0)))
        return {'uid': self.uid, 'canTrade': True, 'accountType': 'SPOT', 'permissions': ['SPOT'],
                'balances': [{'asset': a, 'free': str(q-locked if a == 'BTC' else q), 'locked': str(locked if a == 'BTC' else 0)} for a, q in self.balances.items()]}

    def submit(self, symbol: str, side: str, quantity: str, client_id: str) -> dict[str, Any]:
        self.submissions.append(client_id)
        amount, price = Decimal(quantity), Decimal('101')
        cost, fee = amount * price, amount * price * Decimal('0.002')
        base, quote = symbol.split('/')
        self.balances[base] += amount if side == 'buy' else -amount
        self.balances[quote] += -cost-fee if side == 'buy' else cost-fee
        order = {'symbol': symbol.replace('/', ''), 'orderId': len(self.orders)+1, 'clientOrderId': client_id,
                 'side': side.upper(), 'status': 'FILLED', 'origQty': quantity,
                 'executedQty': quantity, 'cummulativeQuoteQty': str(cost), 'updateTime': int(time.time()*1000)}
        self.orders[client_id] = order
        self.fills.append({'symbol': symbol.replace('/', ''), 'id': len(self.fills)+1, 'orderId': order['orderId'],
                           'price': str(price), 'qty': quantity, 'quoteQty': str(cost), 'commission': str(fee),
                           'commissionAsset': quote, 'isBuyer': side == 'buy', 'time': order['updateTime']})
        return dict(order)

    def query(self, symbol: str, client_id: str) -> dict[str, Any] | None:
        return self.orders.get(client_id)

    def query_order(self, symbol: str, order_id: str) -> dict[str, Any] | None:
        return next((dict(o) for o in self.orders.values() if str(o['orderId']) == order_id and o['symbol'] == symbol.replace('/', '')), None)

    def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return [dict(o) for o in self.orders.values() if (symbol is None or o['symbol'] == symbol.replace('/', '')) and o['status'] in {'NEW', 'PARTIALLY_FILLED'}]

    def trades(self, symbol: str, *, from_id: int = 0, order_id: str | None = None) -> list[dict[str, Any]]:
        return [dict(t) for t in self.fills if t['symbol'] == symbol.replace('/', '') and t['id'] >= from_id
                and (order_id is None or str(t['orderId']) == order_id)]


def test_exchange_fill_projects_actual_provider_cost_and_fee() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        version = sandbox.client.get('/api/status').get_json()['config_version']
        policy = {k: v for k, v in PAPER_INPUTS.items() if k != 'initial_cash'}
        assert sandbox.post('/exchange/setup', data={**policy, 'expected_config_version': str(version)}).status_code == 302
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = SpotFixture()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            response = sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1',
                                                          'csrf_token': sandbox.csrf, 'idempotency_key': 'actual-fill'})
            assert response.status_code == 302
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['operations'][0]['state'] == 'filled'
        assert state['positions'][0]['qty'] == '1'
        assert state['positions'][0]['entry'] == '101.202'
        assert state['fills'][0]['fee_amount'] == '0.202'
        assert state['exchange_account']['balances']['USDT']['total'] == '898.798'
        assert len(provider.submissions) == 1


def test_timeout_after_success_recovers_by_identity_without_resubmission() -> None:
    from execution_worker import ExecutionWorker
    class TimeoutProvider(SpotFixture):
        visible = False
        def submit(self, symbol: str, side: str, quantity: str, client_id: str) -> dict[str, Any]:
            super().submit(symbol, side, quantity, client_id)
            raise TimeoutError('redacted fake uncertain transport')
        def query(self, symbol: str, client_id: str) -> dict[str, Any] | None:
            return super().query(symbol, client_id) if self.visible else None
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        version = sandbox.client.get('/api/status').get_json()['config_version']
        policy = {k: v for k, v in PAPER_INPUTS.items() if k != 'initial_cash'}
        sandbox.post('/exchange/setup', data={**policy, 'expected_config_version': str(version)})
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = TimeoutProvider()
        # Provider clocks use millisecond precision; whole seconds also avoid
        # fabricating a future microsecond when freezing a nanosecond host clock.
        tick = [float(int(time.time()))]
        order = {'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf, 'idempotency_key': 'uncertain-effect'}
        with ExecutionWorker(sandbox.module.state_store, clock=lambda: tick[0], exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data=order)
            owner.process_pending()
            assert sandbox.client.get('/api/status').get_json()['operations'][0]['state'] == 'unknown'
            tick[0] += 20
            sandbox.client.post('/trade', data=order)
            owner.process_pending()
            assert len(provider.submissions) == 1
            assert sandbox.client.get('/api/status').get_json()['positions'] == []
            provider.visible = True
            tick[0] += 15
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['operations'][0]['state'] == 'filled'
        assert state['positions'][0]['qty'] == '1'
        assert len(provider.submissions) == 1


def test_unexplained_provider_balance_reset_keeps_confirmed_facts_in_review() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        version = sandbox.client.get('/api/status').get_json()['config_version']
        policy = {k: v for k, v in PAPER_INPUTS.items() if k != 'initial_cash'}
        sandbox.post('/exchange/setup', data={**policy, 'expected_config_version': str(version)})
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = SpotFixture()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            provider.balances = {'USDT': Decimal('1000'), 'BTC': Decimal('0')}
            owner.reconcile_exchange()
        state = sandbox.client.get('/api/status').get_json()
        assert state['exchange_account']['state'] == 'review_required'
        assert state['exchange_account']['reason'] == 'unexpected_balance_change'
        assert state['positions'][0]['qty'] == '1'
        assert state['fills'][0]['qty'] == '1'


@pytest.mark.parametrize('changes', [
    {'max_allocation_fraction': '0.05'}, {'max_exposure_fraction': '0.05'}, {'risk_fraction': '0.001'},
])
def test_exchange_risk_caps_reject_before_private_submission(changes: dict[str, str]) -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox, **changes)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = SpotFixture()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert provider.submissions == []
        assert state['positions'] == []
        assert state['operations'][0]['state'] == 'rejected'
        assert state['exchange_account']['balances']['USDT']['total'] == '1000'


def test_exchange_quote_change_requires_a_new_review_before_dispatch() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = SpotFixture()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.publish_prices({'BTC/USDT': '102'})
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert provider.submissions == []
        assert state['operations'][0]['state'] == 'rejected'
        assert state['positions'] == []


def test_exchange_entry_cooldown_survives_restart_and_allows_reduction(tmp_path: Path) -> None:
    from execution_worker import ExecutionWorker
    environment = {'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}
    provider = SpotFixture()
    with LegacySandbox(tmp_path, paper_setup=False, environment=environment) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
    with LegacySandbox(tmp_path, paper_setup=False, environment=environment) as restarted:
        restarted.seed_price('BTC/USDT', 100.0)
        with ExecutionWorker(restarted.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            restarted.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': restarted.csrf})
            owner.process_pending()
            assert len(provider.submissions) == 1
            restarted.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'sell', 'qty': '0.5', 'csrf_token': restarted.csrf})
            owner.process_pending()
        state = restarted.client.get('/api/status').get_json()
        assert state['positions'][0]['qty'] == '0.5'
        assert state['operations'][1]['state'] == 'rejected'
        assert state['operations'][2]['state'] == 'filled'


def test_confirmed_exchange_loss_latches_entry_stop_after_rebound() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox, cooldown_seconds='0', drawdown_fraction='0.0001')
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = SpotFixture()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            owner.publish_prices({'BTC/USDT': '102'})
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert len(provider.submissions) == 1
        assert state['operations'][1]['state'] == 'rejected'
        assert state['positions'][0]['qty'] == '1'


def test_partial_cancel_and_old_observations_preserve_fills_and_remainder() -> None:
    from execution_worker import ExecutionWorker
    class PartialProvider(SpotFixture):
        stale = False
        def submit(self, symbol: str, side: str, quantity: str, client_id: str) -> dict[str, Any]:
            order = super().submit(symbol, side, str(Decimal(quantity)*Decimal('0.4')), client_id)
            order.update(origQty=quantity, status='PARTIALLY_FILLED')
            self.orders[client_id] = dict(order)
            return order
        def confirm_cancel(self) -> None:
            for order in self.orders.values():
                if order['side'] == 'BUY':
                    order['status'] = 'CANCELED'
        def query(self, symbol: str, client_id: str) -> dict[str, Any] | None:
            order = super().query(symbol, client_id)
            if self.stale and order:
                return {**order, 'status': 'NEW', 'executedQty': '0', 'cummulativeQuoteQty': '0', 'updateTime': order['updateTime']-1}
            return order
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = PartialProvider()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            state = sandbox.client.get('/api/status').get_json()
            assert state['operations'][0]['state'] == 'partially_filled'
            assert state['positions'][0]['qty'] == '0.4'
            assert state['exchange_account']['balances']['USDT']['reserved'] == '61.206'
            provider.confirm_cancel()
            owner.reconcile_exchange()
            provider.stale = True
            owner.reconcile_exchange()
        state = sandbox.client.get('/api/status').get_json()
        assert state['operations'][0]['state'] == 'canceled'
        assert state['positions'][0]['qty'] == '0.4'
        assert len(state['fills']) == 1
        assert state['exchange_account']['balances']['USDT']['reserved'] == '0'
        assert state['exchange_account']['balances']['USDT']['total'] == '959.5192'


def test_initial_provider_holdings_have_unknown_basis_but_allow_reduction() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = SpotFixture()
        provider.balances['BTC'] = Decimal('2')
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            state = sandbox.client.get('/api/status').get_json()
            assert state['positions'][0]['qty'] == '2'
            assert state['positions'][0]['basis_status'] == 'provider_basis_unknown'
            assert state['open_positions'][0]['entry'] is None
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'sell', 'qty': '0.5', 'csrf_token': sandbox.csrf})
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['positions'][0]['qty'] == '1.5'
        assert state['positions'][0]['basis_status'] == 'provider_basis_unknown'
        assert state['operations'][0]['state'] == 'filled'


def test_pinned_ccxt_precision_adapter_uses_only_explicit_spot_transport() -> None:
    import ccxt
    from exchange_adapter import BinanceSpotOrders
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        transport = SpotFixture()
        exchange = ccxt.binance({'apiKey': 'offline-key', 'secret': 'offline-secret', 'options': {'defaultType': 'spot'}})
        exchange.set_sandbox_mode(True)
        exchange.private_post_orderlist_oco = lambda params: transport.protect('BTC/USDT', params['quantity'], params['belowStopPrice'], params['aboveStopPrice'], params['listClientOrderId'], params['aboveClientOrderId'], params['belowClientOrderId'])
        exchange.private_get_orderlist = lambda params: transport.query_list(params['origClientOrderId'])
        exchange.private_delete_orderlist = lambda params: transport.cancel_list('BTC/USDT', params['listClientOrderId'], params['newClientOrderId'])
        exchange.private_get_account = lambda params: transport.account()
        def full_response(params: dict[str, Any]) -> dict[str, Any]:
            result = transport.submit('BTC/USDT', params['side'].lower(), params['quantity'], params['newClientOrderId'])
            result['transactTime'] = result.pop('updateTime')
            return result
        exchange.private_post_order = full_response
        exchange.private_get_order = lambda params: transport.query_order('BTC/USDT', str(params['orderId'])) if 'orderId' in params else transport.query('BTC/USDT', params['origClientOrderId'])
        exchange.private_get_mytrades = lambda params: transport.trades('BTC/USDT', from_id=params.get('fromId', 0), order_id=str(params['orderId']) if 'orderId' in params else None)
        exchange.private_get_openorders = lambda params: transport.open_orders('BTC/USDT' if 'symbol' in params else None)
        provider = BinanceSpotOrders(sandbox.module.runtime_environment, exchange=exchange)
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            assert 'step' not in exchange.market('BTC/USDT')['limits']['amount']
            for key, qty in [('valid-precision','1'), ('invalid-precision','0.123456')]:
                sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': qty, 'csrf_token': sandbox.csrf, 'idempotency_key': key})
                owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['operations'][0]['state'] == 'filled'
        assert state['operations'][1]['state'] == 'rejected'
        assert len(transport.submissions) == 1
        assert state['positions'][0]['qty'] == '1'
        assert sandbox.network_attempts == 0


def test_entry_budget_deferral_keeps_order_queued_without_reserving_unknown_funds() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = SpotFixture()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            # Reconciliation may use the urgent reserve; entry may not.
            assert owner.exchange_execution is not None
            with owner.exchange_execution.budget._state() as budget:
                for limit in budget['limits']:
                    if limit['kind'] == 'REQUEST_WEIGHT':
                        limit['used'] = limit['limit'] - budget['reserve_weight']
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert provider.submissions == []
        assert state['operations'][0]['state'] == 'queued'
        assert state['exchange_account']['balances']['USDT']['reserved'] == '0'
        assert state['positions'] == []


def test_authoritative_pre_engine_rejection_releases_order_reservation() -> None:
    from execution_worker import ExecutionWorker
    from exchange_adapter import SubmissionRejected
    class RejectingProvider(SpotFixture):
        def submit(self, symbol: str, side: str, quantity: str, client_id: str) -> dict[str, Any]:
            self.submissions.append(client_id)
            raise SubmissionRejected('invalid_market')
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = RejectingProvider()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert len(provider.submissions) == 1
        assert state['operations'][0]['state'] == 'rejected'
        assert state['exchange_account']['balances']['USDT']['reserved'] == '0'
        assert state['exchange_account']['balances']['USDT']['total'] == '1000'
        assert state['fills'] == []


def test_unowned_open_order_outside_scan_universe_blocks_dispatch() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = SpotFixture()
        provider.orders['foreign'] = {'symbol': 'ETHUSDT', 'orderId': 999, 'clientOrderId': 'foreign',
                                      'side': 'BUY', 'status': 'NEW', 'origQty': '1', 'executedQty': '0',
                                      'cummulativeQuoteQty': '0', 'updateTime': int(time.time()*1000)}
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['exchange_account']['state'] == 'review_required'
        assert state['exchange_account']['reason'] == 'unexpected_open_order'
        assert provider.submissions == []
        assert state['positions'] == []
        assert state['operations'][0]['state'] == 'queued'
        assert state['fills'] == []


def test_paginated_trade_history_recovers_every_fill_once() -> None:
    from execution_worker import ExecutionWorker
    class PagedProvider(SpotFixture):
        def submit(self, symbol: str, side: str, quantity: str, client_id: str) -> dict[str, Any]:
            order = super().submit(symbol, side, quantity, client_id)
            original = self.fills.pop()
            self.fills.extend({**original, 'id': i, 'qty': '0.001', 'quoteQty': '0.101', 'commission': '0.000202'}
                              for i in range(1, 1002))
            return order
        def trades(self, symbol: str, *, from_id: int = 0, order_id: str | None = None) -> list[dict[str, Any]]:
            return super().trades(symbol, from_id=from_id, order_id=order_id)[:1000]
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = PagedProvider()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1.001', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            owner.reconcile_exchange()
        state = sandbox.client.get('/api/status').get_json()
        assert state['operations'][0]['state'] == 'filled'
        assert state['positions'][0]['qty'] == '1.001'
        assert state['positions'][0]['entry'] == '101.202'
        assert state['exchange_account']['balances']['USDT']['total'] == '898.696798'
        assert len(state['fills']) == 1001
        assert len(provider.submissions) == 1


def test_reused_provider_fill_identity_with_changed_economics_fails_closed() -> None:
    from execution_worker import ExecutionWorker
    from state_store import StateUnavailable
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = SpotFixture()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            provider.fills[0]['quoteQty'] = '102'
            with pytest.raises(StateUnavailable):
                owner.reconcile_exchange()
        state = sandbox.client.get('/api/status').get_json()
        assert state['fills'][0]['quote_cost'] == '101'
        assert state['positions'][0]['qty'] == '1'
        assert state['exchange_account']['state'] == 'reconciliation_required'
        assert len(provider.submissions) == 1


def test_changed_account_identity_is_reviewed_before_merging_order_facts() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = SpotFixture()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            provider.uid = 'different-offline-account'
            provider.fills[0]['quoteQty'] = '102'
            owner.reconcile_exchange()
        state = sandbox.client.get('/api/status').get_json()
        assert state['exchange_account']['state'] == 'review_required'
        assert state['exchange_account']['reason'] == 'account_identity_mismatch'
        assert state['fills'][0]['quote_cost'] == '101'
        assert state['exchange_account']['balances']['USDT']['total'] == '898.798'


def test_new_unowned_trade_history_requires_review_even_when_balances_match() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = SpotFixture()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            provider.fills.append({'symbol': 'BTCUSDT', 'id': 1, 'orderId': 999, 'price': '100', 'qty': '1',
                                   'quoteQty': '100', 'commission': '0', 'commissionAsset': 'USDT',
                                   'isBuyer': True, 'time': int(time.time()*1000)})
            owner.reconcile_exchange()
        state = sandbox.client.get('/api/status').get_json()
        assert state['exchange_account']['state'] == 'review_required'
        assert state['exchange_account']['reason'] == 'unexpected_trade_history'
        assert state['positions'] == []
        assert state['fills'] == []
        assert provider.submissions == []


@pytest.mark.parametrize(('sizing', 'changes', 'expected'), [
    ('fixed', {}, '0.98'),
    ('risk', {'fee_fraction': '0', 'slippage_fraction': '0', 'risk_fraction': '0.01'}, '3.333'),
])
def test_exchange_server_sizing_uses_reconciled_capital(sizing: str, changes: dict[str, str], expected: str) -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox, **changes)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = SpotFixture()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            response = sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '999999',
                                                          'sizing': sizing, 'csrf_token': sandbox.csrf})
            assert response.status_code == 302
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['positions'][0]['qty'] == expected
        assert state['operations'][0]['requested_qty'] == expected
        assert len(provider.submissions) == 1


def test_idle_exchange_reconciliation_uses_approved_fifteen_second_schedule() -> None:
    from execution_worker import ExecutionWorker
    class CountingProvider(SpotFixture):
        account_reads = 0
        def account(self) -> dict[str, Any]:
            self.account_reads += 1
            return super().account()
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = CountingProvider()
        tick = [float(int(time.time()))]
        with ExecutionWorker(sandbox.module.state_store, clock=lambda: tick[0], exchange_provider=provider) as owner:
            owner.process_pending()
            assert provider.account_reads == 2
            for _ in range(20):
                owner.process_pending()
            assert provider.account_reads == 2
            tick[0] += 15
            owner.process_pending()
            assert provider.account_reads == 4
        assert provider.submissions == []


def test_base_asset_fee_reduces_confirmed_sellable_quantity() -> None:
    from execution_worker import ExecutionWorker
    class BaseFeeProvider(SpotFixture):
        def submit(self, symbol: str, side: str, quantity: str, client_id: str) -> dict[str, Any]:
            order = super().submit(symbol, side, quantity, client_id)
            self.balances['USDT'] += Decimal(self.fills[-1]['commission'])
            self.balances['BTC'] -= Decimal('0.002')
            self.fills[-1].update(commission='0.002', commissionAsset='BTC')
            return order
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = BaseFeeProvider()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['positions'][0]['qty'] == '0.998'
        assert state['exchange_account']['balances']['BTC']['total'] == '0.998'
        assert state['exchange_account']['balances']['BTC']['locked'] == '0.998'
        assert state['protection'][-1]['quantity'] == '0.998'
        assert state['protection'][-1]['state'] == 'active'
        assert state['exchange_account']['balances']['USDT']['total'] == '899'
        assert state['fills'][0]['fee_asset'] == 'BTC'
        assert Decimal(state['positions'][0]['entry']).quantize(Decimal('0.00000001')) == Decimal('101.20240481')


def test_third_asset_fee_updates_its_confirmed_holding_without_invented_conversion() -> None:
    from execution_worker import ExecutionWorker
    class ThirdFeeProvider(SpotFixture):
        def submit(self, symbol: str, side: str, quantity: str, client_id: str) -> dict[str, Any]:
            order = super().submit(symbol, side, quantity, client_id)
            self.balances['USDT'] += Decimal(self.fills[-1]['commission'])
            self.balances['BNB'] -= Decimal('0.001')
            self.fills[-1].update(commission='0.001', commissionAsset='BNB')
            return order
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = ThirdFeeProvider()
        provider.balances.update(BTC=Decimal('2'), BNB=Decimal('1'))
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/close', data={'symbol': 'BTC/USDT', 'qty': '0.5', 'csrf_token': sandbox.csrf})
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        positions = {p['symbol']: p for p in state['positions']}
        assert positions['BTC/USDT']['qty'] == '1.5'
        assert positions['BNB/USDT']['qty'] == '0.999'
        assert positions['BNB/USDT']['basis_status'] == 'provider_basis_unknown'
        assert state['exchange_account']['balances']['BNB']['total'] == '0.999'
        assert state['exchange_account']['balances']['USDT']['total'] == '1050.5'
        assert state['fills'][0]['fee_asset'] == 'BNB'
        assert state['fills'][0]['fee_amount'] == '0.001'


def test_unresolved_same_account_namespace_cannot_be_bypassed_by_exchange_mode() -> None:
    from execution_worker import ExecutionWorker
    from state_store import StateStore
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        store = sandbox.module.state_store
        other = StateStore(store.path.parent, store.account, store.venue, 'paper')
        other.provision(sandbox.module.environment_defaults({}))
        operation = other.enqueue('trade', 'BTC/USDT', 'buy', '1')
        with other.connection(write=True) as connection:
            connection.execute("UPDATE intents SET state='unknown' WHERE operation_id=?", (operation,))
        provider = SpotFixture()
        with ExecutionWorker(store, exchange_provider=provider) as owner:
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['exchange_account']['state'] == 'review_required'
        assert state['exchange_account']['reason'] == 'other_namespace_unresolved'
        assert provider.submissions == []
        assert state['positions'] == []


@pytest.mark.parametrize('after_success', [False, True])
def test_crashed_dispatch_recovers_before_any_new_intent(after_success: bool) -> None:
    from execution_worker import ExecutionWorker
    class ProcessCrash(BaseException):
        pass
    class CrashingProvider(SpotFixture):
        def submit(self, symbol: str, side: str, quantity: str, client_id: str) -> dict[str, Any]:
            if after_success:
                super().submit(symbol, side, quantity, client_id)
            raise ProcessCrash()
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = CrashingProvider()
        with pytest.raises(ProcessCrash):
            with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
                owner.reconcile_exchange()
                sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1',
                                                      'csrf_token': sandbox.csrf, 'idempotency_key': 'crashing-intent'})
                owner.process_pending()
        before = sandbox.client.get('/api/status').get_json()
        assert before['operations'][0]['state'] == 'dispatching'
        assert before['positions'] == []
        sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1',
                                              'csrf_token': sandbox.csrf, 'idempotency_key': 'new-intent'})
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as recovered:
            recovered.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert len(provider.submissions) == (1 if after_success else 0)
        assert state['operations'][0]['state'] == ('filled' if after_success else 'dispatching')
        assert state['operations'][1]['state'] == ('rejected' if after_success else 'queued')
        if after_success:
            assert state['positions'][0]['qty'] == '1'
        else:
            assert state['positions'] == []


def test_failed_fill_transaction_preserves_dispatch_identity_for_recovery() -> None:
    from execution_worker import ExecutionWorker
    from state_store import StateUnavailable
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = SpotFixture()
        store = sandbox.module.state_store
        with store.connection(write=True) as connection:
            connection.execute("CREATE TRIGGER offline_write_failure BEFORE INSERT ON fills BEGIN SELECT RAISE(ABORT,'offline write failure'); END")
        with ExecutionWorker(store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            with pytest.raises(StateUnavailable):
                owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['operations'][0]['state'] == 'dispatching'
        assert state['positions'] == []
        assert state['fills'] == []
        with store.connection(write=True) as connection:
            connection.execute('DROP TRIGGER offline_write_failure')
        with ExecutionWorker(store, exchange_provider=provider) as recovered:
            recovered.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['operations'][0]['state'] == 'filled'
        assert state['positions'][0]['qty'] == '1'
        assert len(provider.submissions) == 1


def test_entry_pause_blocks_buys_but_keeps_authenticated_reduction_available() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = SpotFixture()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            assert sandbox.client.post('/exchange/entries', data={'action': 'pause', 'csrf_token': sandbox.csrf}).status_code == 302
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            sandbox.client.post('/close', data={'symbol': 'BTC/USDT', 'qty': '0.5', 'csrf_token': sandbox.csrf})
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['entry_control']['paused'] is True
        assert state['operations'][1]['error_class'] == 'entry_paused'
        assert state['operations'][2]['state'] == 'filled'
        assert state['positions'][0]['qty'] == '0.5'
        assert len(provider.submissions) == 2
        assert sandbox.client.post('/exchange/entries', data={'action': 'resume', 'csrf_token': sandbox.csrf}).status_code == 302
        assert sandbox.client.get('/api/status').get_json()['entry_control']['paused'] is False


@pytest.mark.parametrize('quantity', ['0.0005', '0.01', '20'])
def test_exchange_lot_notional_and_free_asset_rejections_have_no_private_effect(quantity: str) -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = SpotFixture()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': quantity, 'csrf_token': sandbox.csrf})
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['operations'][0]['state'] == 'rejected'
        assert state['exchange_account']['balances']['USDT']['total'] == '1000'
        assert state['positions'] == []
        assert provider.submissions == []


def test_other_venue_provider_cannot_publish_into_this_namespace() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = SpotFixture()
        provider.venue = 'mainnet'
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
        state = sandbox.client.get('/api/status').get_json()
        assert state['exchange_account']['state'] == 'review_required'
        assert state['exchange_account']['reason'] == 'venue_identity_mismatch'
        assert state['exchange_account']['balances'] == {}
        assert state['positions'] == []
        assert provider.submissions == []


def test_malformed_fill_after_submit_cannot_be_classified_as_rejected() -> None:
    from execution_worker import ExecutionWorker
    from state_store import StateUnavailable
    class MalformedProvider(SpotFixture):
        def submit(self, symbol: str, side: str, quantity: str, client_id: str) -> dict[str, Any]:
            order = super().submit(symbol, side, quantity, client_id)
            self.fills[-1]['commission'] = 'NaN'
            return order
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = MalformedProvider()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            with pytest.raises(StateUnavailable):
                owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['operations'][0]['state'] == 'dispatching'
        assert state['exchange_account']['balances']['USDT']['reserved'] == '102.01'
        assert state['exchange_account']['state'] == 'reconciliation_required'
        assert state['fills'] == []
        provider.fills[-1]['commission'] = '0.202'
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as recovered:
            recovered.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['operations'][0]['state'] == 'filled'
        assert state['positions'][0]['qty'] == '1'
        assert len(provider.submissions) == 1


def test_sparse_ids_and_delayed_lower_trade_ids_cannot_escape_review() -> None:
    from execution_worker import ExecutionWorker
    class SparseProvider(SpotFixture):
        def submit(self, symbol: str, side: str, quantity: str, client_id: str) -> dict[str, Any]:
            result = super().submit(symbol, side, quantity, client_id)
            self.fills[-1]['id'] = 200000
            return result
    with LegacySandbox(paper_setup=False, environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        configure(sandbox)
        sandbox.seed_price('BTC/USDT', 100.0)
        provider = SparseProvider()
        with ExecutionWorker(sandbox.module.state_store, exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            assert sandbox.client.get('/api/status').get_json()['exchange_account']['state'] == 'ready'
            for identity, buyer in [(100000, True), (100001, False)]:
                provider.fills.append({'symbol': 'BTCUSDT', 'id': identity, 'orderId': identity, 'price': '100', 'qty': '1',
                                       'quoteQty': '100', 'commission': '0', 'commissionAsset': 'USDT',
                                       'isBuyer': buyer, 'time': int(time.time()*1000)})
            owner.reconcile_exchange()
        state = sandbox.client.get('/api/status').get_json()
        assert state['exchange_account']['state'] == 'review_required'
        assert state['exchange_account']['reason'] == 'unexpected_trade_history'
        assert len(state['fills']) == 1
        assert state['positions'][0]['qty'] == '1'
        assert len(provider.submissions) == 1
