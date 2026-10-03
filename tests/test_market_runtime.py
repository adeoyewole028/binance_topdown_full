"""Freshness/config/supervision through the approved operator/owned-provider seam."""
from tests.legacy_sandbox import LegacySandbox
import pytest
from datetime import datetime, timezone
from typing import Any


class MarketFixture:
    """Observable public provider with completed fixtures; no transport."""
    def __init__(self, now: float) -> None:
        self.now = now
        self.calls: list[tuple[str, str]] = []
        self.bad_candles = False
        self.fail_prices = False
        self.symbols = [{'symbol': 'BTCUSDT', 'baseAsset': 'BTC', 'quoteAsset': 'USDT', 'status': 'TRADING', 'isSpotTradingAllowed': True, 'orderTypes': ['MARKET'],
                         'filters': [{'filterType': 'LOT_SIZE', 'minQty': '0.001', 'maxQty': '100000', 'stepSize': '0.001'},
                                     {'filterType': 'MIN_NOTIONAL', 'minNotional': '5', 'applyToMarket': True, 'avgPriceMins': 0}]}]
        self.prices = {'BTC/USDT': {'last': 100.0, 'quoteVolume': 20000000.0}}
        self.ticker_calls = 0
        self.metadata_calls = 0

    def metadata(self) -> dict[str, Any]:
        self.metadata_calls += 1
        return {'serverTime': int(self.now * 1000), 'rateLimits': [
            {'rateLimitType': 'REQUEST_WEIGHT', 'interval': 'MINUTE', 'intervalNum': 1, 'limit': 1000}],
            'symbols': self.symbols}

    def reference_price(self, symbol: str) -> dict[str, Any]:
        return {'referencePrice': None}

    def average_price(self, symbol: str) -> dict[str, Any]:
        return {'mins': 5, 'price': str(self.prices[symbol]['last'])}

    def tickers(self, symbols: list[str] | None = None) -> dict[str, Any]:
        self.ticker_calls += 1
        if self.fail_prices:
            raise TimeoutError('offline transient failure')
        return {s: p for s, p in self.prices.items() if symbols is None or s in symbols}

    def candles(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        self.calls.append((symbol, timeframe))
        duration = {'4h': 14400, '1h': 3600, '15m': 900}[timeframe]
        boundary = int(self.now // duration) * duration
        rows = [[(boundary - (limit - index) * duration) * 1000, 100.0, 102.0, 98.0, 100.0, 10.0]
                for index in range(limit)]
        if self.bad_candles:
            rows[-1][0] = rows[-2][0]  # Invalid duplicate timestamp.
        return rows


class ControlledClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 10, 3, 12, tzinfo=timezone.utc).timestamp()
    def __call__(self) -> float:
        return self.value


def test_invalid_configuration_is_rejected_atomically_instead_of_partially_saved() -> None:
    with LegacySandbox() as sandbox:
        before = sandbox.client.get('/api/status').get_json()['config']
        response = sandbox.post('/update-config', data={'risk': '-2', 'max_positions': '3'})
        assert response.status_code == 422
        assert sandbox.client.get('/api/status').get_json()['config'] == before


@pytest.mark.parametrize(('field', 'value'), [
    ('risk', 'NaN'), ('risk', 'inf'), ('risk', '0'), ('risk', '1.01'),
    ('max_positions', '1.5'), ('max_positions', '0'), ('cooldown', '-1'),
    ('near_ema_pct', '-0.1'), ('stop_loss_pct', '1'), ('take_profit_pct', '0'),
    ('rsi_threshold', '101'), ('rsi_exec_threshold', '80'), ('min_quote_vol', '-1'),
    ('equity_proxy', '-1'), ('base_quote', '../USDT'), ('use_dynamic_universe', 'sometimes'),
])
def test_invalid_settings_leave_the_complete_version_unchanged(field: str, value: str) -> None:
    with LegacySandbox() as sandbox:
        before = sandbox.client.get('/api/status').get_json()
        response = sandbox.post('/update-config', data={'cooldown': '90', field: value})
        assert response.status_code == 422
        after = sandbox.client.get('/api/status').get_json()
        assert after['config'] == before['config']
        assert after['config_version'] == before['config_version']


def test_stale_settings_review_cannot_overwrite_a_newer_version() -> None:
    with LegacySandbox() as sandbox:
        version = str(sandbox.client.get('/api/status').get_json()['config_version'])
        sandbox.post('/update-config', data={'risk': '0.02', 'expected_config_version': version})
        response = sandbox.post('/update-config', data={'risk': '0.03', 'expected_config_version': version})
        assert response.status_code == 409
        assert sandbox.client.get('/api/status').get_json()['config']['risk'] == 0.02


def test_stale_price_never_becomes_a_confirmed_paper_fill() -> None:
    from execution_worker import ExecutionWorker
    with LegacySandbox() as sandbox:
        sandbox.seed_price('BTC/USDT', 100)
        sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
        with ExecutionWorker(sandbox.module.state_store, clock=lambda: 4102444800.0) as owner:
            owner.process_pending()
        state = sandbox.client.get('/api/status').get_json()
        assert state['positions'] == []
        assert state['operations'][0]['state'] == 'waiting_for_market_data'


def test_duplicate_candles_degrade_scan_without_advancing_last_completed_scan() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock = ControlledClock()
    provider = MarketFixture(clock())
    with LegacySandbox() as sandbox:
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            sandbox.module.initialize_runtime(clock=clock)
            scanner = MarketRuntime(owner, provider, clock=clock)
            scanner.scan_once()
            first = sandbox.client.get('/api/status').get_json()['market_health']
            assert first['state'] == 'ready'
            provider.bad_candles = True
            clock.value += 900
            provider.now = clock()
            scanner.scan_once()
            health = sandbox.client.get('/api/status').get_json()['market_health']
            assert health['state'] == 'degraded'
            assert health['reason'] == 'data_quality'
            assert health['last_completed_scan'] == first['last_completed_scan']


def test_scanner_staleness_blocks_dispatch_even_with_a_new_price() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock = ControlledClock()
    provider = MarketFixture(clock())
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            scanner = MarketRuntime(owner, provider, clock=clock)
            scanner.scan_once()
            clock.value += 600
            provider.now = clock()
            scanner.refresh_prices()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            state = sandbox.client.get('/api/status').get_json()
            assert state['market_health']['execution_ready'] is False
            assert state['market_health']['reason'] == 'scanner_stale'
            assert state['positions'] == []
            assert state['operations'][0]['state'] == 'waiting_for_market_data'


def test_completed_candle_cache_survives_owner_restart_and_reuses_slow_timeframes() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock = ControlledClock()
    provider = MarketFixture(clock())
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            MarketRuntime(owner, provider, clock=clock).scan_once()
        assert len(provider.calls) == 3
        clock.value += 300
        provider.now = clock()
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            scanner = MarketRuntime(owner, provider, clock=clock)
            scanner.scan_once()
            assert len(provider.calls) == 3
            clock.value += 600
            provider.now = clock()
            scanner.scan_once()
            assert provider.calls[3:] == [('BTC/USDT', '15m')]
            state = sandbox.client.get('/api/status').get_json()
            assert state['market_health']['execution_ready'] is True


def test_independent_refresh_cadence_does_not_repeat_full_scans_or_catch_up_bursts() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock = ControlledClock()
    provider = MarketFixture(clock())
    reconciliations = []
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            scanner = MarketRuntime(owner, provider, clock=clock, reconcile=lambda: reconciliations.append(clock()))
            scanner.step()
            clock.value += 15
            provider.now = clock()
            scanner.step()
            assert provider.ticker_calls == 1
            clock.value += 15
            provider.now = clock()
            scanner.step()
            assert provider.ticker_calls == 2
            clock.value += 270
            provider.now = clock()
            scanner.step()
            assert provider.metadata_calls == 2
            assert provider.ticker_calls == 3
            assert len(reconciliations) == 4
            assert len(provider.calls) == 3


def test_failed_price_refresh_preserves_age_and_expires_dispatch() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock = ControlledClock()
    provider = MarketFixture(clock())
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            scanner = MarketRuntime(owner, provider, clock=clock)
            scanner.scan_once()
            initial = sandbox.client.get('/api/status').get_json()['market_prices']['BTC/USDT']['fetched_at']
            provider.fail_prices = True
            clock.value += 60
            scanner.refresh_prices()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            state = sandbox.client.get('/api/status').get_json()
            assert state['market_prices']['BTC/USDT']['fetched_at'] == initial
            assert state['market_prices']['BTC/USDT']['fresh'] is False
            assert state['market_health']['execution_ready'] is False
            assert state['market_health']['reason'] == 'price_stale'
            assert state['positions'] == []


def test_provider_rate_limit_wait_is_shared_across_scanner_restarts() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    from request_budget import ProviderFailure
    clock = ControlledClock()
    class LimitedProvider(MarketFixture):
        attempts = 0
        def tickers(self, symbols: list[str] | None = None) -> dict[str, Any]:
            self.attempts += 1
            raise ProviderFailure('rate_limit', retry_after=120)
    provider = LimitedProvider(clock())
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            MarketRuntime(owner, provider, clock=clock).scan_once()
            assert sandbox.client.get('/api/status').get_json()['market_health']['reason'] == 'rate_limit'
            for _ in range(4):
                clock.value += 1
                MarketRuntime(owner, provider, clock=clock).scan_once()
            assert provider.attempts == 1
            clock.value += 120
            provider.now = clock()
            MarketRuntime(owner, provider, clock=clock).scan_once()
            assert provider.attempts == 2


def test_research_cannot_consume_the_reserved_reconciliation_capacity() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    from request_budget import BudgetDeferred
    clock = ControlledClock()
    class RawLimitedProvider(MarketFixture):
        def metadata(self) -> dict[str, Any]:
            result = super().metadata()
            result['rateLimits'].append({'rateLimitType': 'RAW_REQUESTS', 'interval': 'MINUTE', 'intervalNum': 1, 'limit': 16})
            return result
    provider = RawLimitedProvider(clock())
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            scanner = MarketRuntime(owner, provider, clock=clock)
            scanner.scan_once()
            count = 0
            for _ in range(100):
                try:
                    scanner.budget.request('research', 80, provider.tickers)
                    count += 1
                except BudgetDeferred:
                    break
            assert count == 1
            replacement = MarketRuntime(owner, provider, clock=clock)
            result = replacement.budget.request('reconciliation', 80, provider.tickers)
            assert result['BTC/USDT']['last'] == 100


def test_quote_and_universe_filters_preserve_held_and_pending_coverage() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock = ControlledClock()
    provider = MarketFixture(clock())
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            scanner = MarketRuntime(owner, provider, clock=clock, whitelist={'ETH/USDC'}, blacklist={'DOGE/USDC'})
            # Establish a legitimate held symbol before changing quote policy.
            initial = MarketRuntime(owner, provider, clock=clock)
            initial.scan_once()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            assert sandbox.client.get('/api/status').get_json()['positions'][0]['symbol'] == 'BTC/USDT'
            for base in ('ETH', 'DOGE'):
                provider.symbols.append({'symbol': base + 'USDC', 'baseAsset': base, 'quoteAsset': 'USDC',
                                         'status': 'TRADING', 'isSpotTradingAllowed': True})
                provider.prices[base + '/USDC'] = {'last': 100.0, 'quoteVolume': 20000000.0}
            sandbox.post('/update-config', data={'base_quote': 'USDC', 'min_quote_vol': '999999999'})
            provider.symbols.append({'symbol': 'DOGEUSDT', 'baseAsset': 'DOGE', 'quoteAsset': 'USDT', 'status': 'TRADING', 'isSpotTradingAllowed': True})
            provider.prices['DOGE/USDT'] = {'last': 100.0, 'quoteVolume': 20000000.0}
            owner.publish_prices({'DOGE/USDT': 100})
            sandbox.client.post('/trade', data={'symbol': 'DOGE/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            scanner.scan_once()
            state = sandbox.client.get('/api/status').get_json()
            assert state['universe'] == ['ETH/USDC']
            assert state['market_health']['coverage'] == ['BTC/USDT', 'DOGE/USDT', 'ETH/USDC']
            assert state['market_health']['execution_ready'] is True


def test_new_candle_close_requires_updated_history_before_dispatch() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock = ControlledClock()
    clock.value += 840  # 12:14; latest completed execution candle opened at 11:45.
    provider = MarketFixture(clock())
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            scanner = MarketRuntime(owner, provider, clock=clock)
            scanner.scan_once()
            clock.value += 60
            provider.now = clock()
            scanner.refresh_prices()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            state = sandbox.client.get('/api/status').get_json()
            assert state['market_health']['reason'] == 'candle_stale'
            assert state['positions'] == []
            scanner.scan_once()
            owner.process_pending()
            assert sandbox.client.get('/api/status').get_json()['positions'][0]['qty'] == '1'


def test_public_spot_adapter_and_existing_evaluator_use_only_owned_completed_data() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    from market_adapter import BinancePublicMarket
    clock = ControlledClock()
    provider = MarketFixture(clock())
    class SpotTransport:
        last_response_headers: dict[str, Any] | None = None
        calls: list[str]
        def __init__(self) -> None:
            self.calls = []
        def public_get_exchangeinfo(self, params: dict[str, Any]) -> dict[str, Any]:
            self.calls.append('exchangeInfo')
            return provider.metadata()
        def public_get_time(self, params: dict[str, Any]) -> dict[str, Any]:
            self.calls.append('time')
            return {'serverTime': int(clock() * 1000)}
        def public_get_ticker_24hr(self, params: dict[str, Any]) -> list[dict[str, Any]]:
            self.calls.append('ticker24hr')
            return [{'symbol': 'BTCUSDT', 'lastPrice': str(provider.prices['BTC/USDT']['last']), 'quoteVolume': '20000000'}]
        def public_get_referenceprice(self, params: dict[str, Any]) -> dict[str, Any]:
            self.calls.append('referencePrice')
            return {'referencePrice': None}
        def public_get_klines(self, params: dict[str, Any]) -> list[list[float]]:
            self.calls.append('klines')
            return provider.candles('BTC/USDT', params['interval'], params['limit'])
    transport = SpotTransport()
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            scanner = MarketRuntime(owner, BinancePublicMarket(transport), clock=clock,
                                    evaluate=sandbox.module.evaluate_market_snapshot)
            scanner.scan_once()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            clock.value += 300
            provider.now = clock()
            provider.prices['BTC/USDT']['last'] = 120
            sandbox.post('/update-config', data={'take_profit_pct': '0.5'})
            scanner.scan_once()
            state = sandbox.client.get('/api/status').get_json()
            assert state['positions'][0]['qty'] == '1'
            assert 'take_profit' in state['exit_signals'][0]['reasons']
            assert state['position_rules']['BTC/USDT']['config_version'] == 2
            assert state['market_health']['execution_ready'] is True
            assert transport.calls[:2] == ['exchangeInfo', 'time']
            assert sandbox.network_attempts == 0


@pytest.mark.parametrize('damage', ['gap', 'ohlc', 'nonfinite', 'warmup', 'unsorted'])
def test_invalid_completed_history_is_not_execution_ready(damage: str) -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    class DamagedProvider(MarketFixture):
        def candles(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
            rows = super().candles(symbol, timeframe, limit)
            if damage == 'gap':
                del rows[-2]
            elif damage == 'ohlc':
                rows[-1][2] = 90
            elif damage == 'nonfinite':
                rows[-1][4] = float('nan')
            elif damage == 'warmup':
                rows = rows[-3:]
            elif damage == 'unsorted':
                rows = rows[::-1]
            return rows
    clock = ControlledClock()
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            MarketRuntime(owner, DamagedProvider(clock()), clock=clock).scan_once()
            state = sandbox.client.get('/api/status').get_json()
            assert state['market_health']['execution_ready'] is False
            assert state['market_health']['reason'] == 'data_quality'
            assert state['last_scan'] == ''


def test_forming_candle_cannot_reach_the_evaluator() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime, CachedMarketView
    clock = ControlledClock()
    class FormingProvider(MarketFixture):
        def candles(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
            rows = super().candles(symbol, timeframe, limit)
            duration = {'4h': 14400, '1h': 3600, '15m': 900}[timeframe]
            boundary = int(clock() // duration) * duration
            rows.append([boundary * 1000, 999, 1000, 998, 999, 10])
            return rows
    def evaluate(view: CachedMarketView, universe: list[str], config: dict[str, Any]) -> dict[str, Any]:
        return {'signals': [{'symbol': 'BTC/USDT', 'fixture_close': view.fetch_ohlcv('BTC/USDT', timeframe='15m', limit=100)[-1][4]}]}
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            MarketRuntime(owner, FormingProvider(clock()), clock=clock, evaluate=evaluate).scan_once()
            state = sandbox.client.get('/api/status').get_json()
            assert state['market_health']['execution_ready'] is True
            assert state['signals'][0]['fixture_close'] == 100


def test_auth_failure_requires_explicit_review_and_has_no_retry_storm() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    from request_budget import ProviderFailure
    clock = ControlledClock()
    class UnauthorizedProvider(MarketFixture):
        attempts = 0
        denied = True
        def metadata(self) -> dict[str, Any]:
            self.attempts += 1
            if self.denied:
                raise ProviderFailure('auth')
            return super().metadata()
    provider = UnauthorizedProvider(clock())
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            for _ in range(4):
                MarketRuntime(owner, provider, clock=clock).scan_once()
                clock.value += 300
            state = sandbox.client.get('/api/status').get_json()
            assert provider.attempts == 1
            assert state['market_health']['reason'] == 'auth'
            assert state['market_health']['execution_ready'] is False
            provider.denied = False
            provider.now = clock()
            scanner = MarketRuntime(owner, provider, clock=clock)
            scanner.budget.resume(owner, actor='operator')
            MarketRuntime(owner, provider, clock=clock).scan_once()
            assert sandbox.client.get('/api/status').get_json()['market_health']['execution_ready'] is True


def test_protected_symbol_remains_covered_after_the_position_and_universe_change() -> None:
    import sqlite3
    from contextlib import closing
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock = ControlledClock()
    provider = MarketFixture(clock())
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            scanner = MarketRuntime(owner, provider, clock=clock)
            scanner.scan_once()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            operation = sandbox.client.get('/api/status').get_json()['operations'][0]['operation_id']
            # Durable in-flight protection fixture, not a claim of BT07 activation.
            with closing(sqlite3.connect(sandbox.directory / 'trading.sqlite3')) as fixture:
                fixture.execute('INSERT INTO protection_groups(namespace,group_id,symbol,operation_id,quantity,state) VALUES (?,?,?,?,?,?)',
                                (sandbox.module.state_store.namespace, 'fixture-protection', 'BTC/USDT', operation, '1', 'installing'))
                fixture.commit()
            sandbox.client.post('/close', data={'symbol': 'BTC/USDT', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            assert sandbox.client.get('/api/status').get_json()['positions'] == []
            provider.symbols.append({'symbol': 'ETHUSDC', 'baseAsset': 'ETH', 'quoteAsset': 'USDC',
                                     'status': 'TRADING', 'isSpotTradingAllowed': True})
            provider.prices['ETH/USDC'] = {'last': 100.0, 'quoteVolume': 20000000.0}
            sandbox.post('/update-config', data={'base_quote': 'USDC'})
            scanner.scan_once()
            state = sandbox.client.get('/api/status').get_json()
            assert state['universe'] == ['ETH/USDC']
            assert state['market_health']['coverage'] == ['BTC/USDT', 'ETH/USDC']


def test_shared_ip_budget_is_not_reset_by_a_second_account_namespace() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    from request_budget import RequestBudget, BudgetDeferred
    from state_store import StateStore
    clock = ControlledClock()
    provider = MarketFixture(clock())
    with LegacySandbox() as sandbox:
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            scanner = MarketRuntime(owner, provider, clock=clock)
            scanner.scan_once()
            for _ in range(9):
                scanner.budget.request('research', 80, provider.tickers)
            other = StateStore(sandbox.directory, 'other-offline-account', 'testnet', 'paper')
            other.provision(sandbox.module.environment_defaults(sandbox.module.os.environ))
            shared = RequestBudget(other, clock=clock)
            with pytest.raises(BudgetDeferred):
                shared.request('research', 80, provider.tickers)
            assert shared.request('reconciliation', 80, provider.tickers)['BTC/USDT']['last'] == 100


def test_mixed_entry_configurations_require_review_instead_of_guessed_exit_rules() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock = ControlledClock()
    provider = MarketFixture(clock())
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            scanner = MarketRuntime(owner, provider, clock=clock, evaluate=sandbox.module.evaluate_market_snapshot)
            scanner.scan_once()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            sandbox.post('/update-config', data={'take_profit_pct': '0.5'})
            scanner.scan_once()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            provider.prices['BTC/USDT']['last'] = 120
            scanner.scan_once()
            state = sandbox.client.get('/api/status').get_json()
            assert state['positions'][0]['qty'] == '2'
            assert state['position_rules']['BTC/USDT']['state'] == 'review_required'
            assert state['exit_signals'] == []


def test_slow_history_fetch_never_freshens_the_earlier_ticker_observation() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock = ControlledClock()
    class SlowProvider(MarketFixture):
        def candles(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
            rows = super().candles(symbol, timeframe, limit)
            clock.value += 21
            return rows
    provider = SlowProvider(clock())
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store, clock=clock) as owner:
            scanner = MarketRuntime(owner, provider, clock=clock)
            scanner.scan_once()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            state = sandbox.client.get('/api/status').get_json()
            assert state['market_prices']['BTC/USDT']['age_seconds'] == 63
            assert state['market_health']['execution_ready'] is False
            assert state['positions'] == []
            scanner.refresh_prices()
            sandbox.client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': sandbox.csrf})
            owner.process_pending()
            assert sandbox.client.get('/api/status').get_json()['positions'][0]['qty'] == '1'


def test_response_crossing_budget_window_preserves_reported_usage_and_urgent_headroom() -> None:
    from request_budget import RequestBudget, BudgetDeferred
    clock = ControlledClock()
    clock.value += 59
    provider = MarketFixture(clock())
    with LegacySandbox() as sandbox:
        budget = RequestBudget(sandbox.module.state_store, clock=clock)
        budget.configure(provider.metadata())
        def delayed_response() -> dict[str, Any]:
            clock.value += 2
            return provider.tickers()
        budget.request('research', 80, delayed_response)
        budget.observe({'X-MBX-USED-WEIGHT-1M': '900'})
        with pytest.raises(BudgetDeferred):
            budget.request('research', 80, provider.tickers)
        assert budget.request('reconciliation', 80, provider.tickers)['BTC/USDT']['last'] == 100


@pytest.mark.parametrize('timing_known', [False, True])
def test_uncertain_server_clock_cannot_discard_response_usage_at_rollover(timing_known: bool) -> None:
    from request_budget import RequestBudget, BudgetDeferred
    clock = ControlledClock()
    clock.value += 59
    provider = MarketFixture(clock() - 2)
    with LegacySandbox() as sandbox:
        budget = RequestBudget(sandbox.module.state_store, clock=clock)
        budget.configure(provider.metadata(), request_started=clock() - 2 if timing_known else None)
        def delayed_response() -> dict[str, Any]:
            clock.value += 2
            return provider.tickers()
        budget.request('research', 80, delayed_response)
        budget.observe({'X-MBX-USED-WEIGHT-1M': '900'})
        clock.value += 2
        with pytest.raises(BudgetDeferred):
            budget.request('research', 80, provider.tickers)
        assert budget.request('reconciliation', 80, provider.tickers)['BTC/USDT']['last'] == 100
