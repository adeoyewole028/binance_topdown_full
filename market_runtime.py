"""Owned public market observations; only completed, validated bars are usable."""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import math
import time
import random
from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from automation import Automation
from decimal import Decimal, InvalidOperation

from execution_worker import ExecutionWorker
from operations_policy import OperationsPolicy
from request_budget import ProviderFailure, RequestBudget, BudgetDeferred


class MarketProvider(Protocol):
    def metadata(self) -> dict[str, Any]: ...
    def tickers(self, symbols: list[str] | None = None) -> dict[str, Any]: ...
    def candles(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]: ...
    def reference_price(self, symbol: str) -> dict[str, Any]: ...
    def average_price(self, symbol: str) -> dict[str, Any]: ...


class DataQualityError(RuntimeError):
    pass


INTERVALS = {'1d': (86400, 300), '4h': (14400, 200), '1h': (3600, 200), '15m': (900, 100)}


def iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


class CachedMarketView:
    """Legacy evaluator-facing public shape; deliberately has no transport."""
    def __init__(self, venue: str, cache: dict[str, Any], prices: dict[str, Any], *, context: dict[str, Any] | None = None) -> None:
        self.venue, self.cache, self.prices = venue, cache, prices
        self.context = context or {}

    def fetch_ohlcv(self, symbol: str, *, timeframe: str, limit: int) -> list[list[float]]:
        rows: list[list[float]] = self.cache[f'{self.venue}:{symbol}:{timeframe}']['rows']
        if len(rows) < limit:
            raise DataQualityError('Insufficient cached warmup')
        return rows[-limit:]

    def fetch_ticker(self, symbol: str) -> dict[str, float]:
        return {'last': float(self.prices[symbol]['last'])}


def validated_bars(incoming: Any, timeframe: str, server_time: float, *, required: int | None = None) -> list[list[float]]:
    duration, default_required = INTERVALS[timeframe]
    required = default_required if required is None else required
    boundary = int(server_time // duration) * duration
    rows: list[list[float]] = []
    previous: float | None = None
    for raw in incoming:
        if len(raw) != 6 or any(isinstance(x, bool) or not math.isfinite(float(x)) for x in raw):
            raise DataQualityError('Malformed candle')
        stamp, opening, high, low, closing, volume = [float(x) for x in raw]
        if stamp % (duration * 1000) or (previous is not None and stamp - previous != duration * 1000):
            raise DataQualityError('Unsorted, duplicate or gapped candle')
        previous = stamp
        if min(opening, high, low, closing) <= 0 or volume < 0 or not low <= min(opening, closing) <= max(opening, closing) <= high:
            raise DataQualityError('Invalid OHLC')
        if stamp / 1000 + duration <= server_time:
            rows.append([stamp, opening, high, low, closing, volume])
    if len(rows) < required or rows[-1][0] != (boundary - duration) * 1000:
        raise DataQualityError('Insufficient warmup or stale history')
    return rows[-required:]


class MarketRuntime:
    def __init__(self, owner: ExecutionWorker, provider: MarketProvider, *,
                 clock: Callable[[], float] = time.time, whitelist: set[str] | None = None,
                 blacklist: set[str] | None = None, reconcile: Callable[[], None] | None = None,
                 evaluate: Callable[[CachedMarketView, list[str], dict[str, Any]], dict[str, Any]] | None = None,
                 automation: Automation | None = None) -> None:
        self.owner, self.provider, self.clock = owner, provider, clock
        self.snapshot = owner.store.scan()
        self.cache: dict[str, Any] = self.snapshot.get('market_cache', {})
        urgent_symbols = max(1, len(owner.store.market_coverage()))
        self.budget = RequestBudget(owner.store, clock=clock, reserve_weight=101 + 10 * urgent_symbols,
                                    reserve_requests=3 + 5 * urgent_symbols)
        self.whitelist, self.blacklist = whitelist or set(), blacklist or set()
        self.reconcile = reconcile
        self.evaluate = evaluate
        self.automation = automation
        self.next_scan = self.next_prices = self.next_reconcile = 0.0

    def _request(self, lane: str, weight: int, call: Callable[[], Any], *,
                 raw_requests: int = 1, discovery: bool = False) -> Any:
        attempted = False
        def attempt() -> Any:
            nonlocal attempted
            attempted = True
            return call()
        try:
            return self.budget.request(lane, weight, attempt, raw_requests=raw_requests, discovery=discovery)
        finally:
            if attempted:
                self.budget.observe(getattr(self.provider, 'headers', {}))

    def step(self) -> None:
        now = self.clock()
        if now >= self.next_reconcile:
            if self.reconcile is not None:
                self.reconcile()
            self.next_reconcile = now + self.owner.policy.reconciliation_interval
        if now >= self.next_scan:
            self.scan_once()
            self.next_scan = now + self.owner.policy.scan_interval
            health = self.snapshot.get('market_health', {})
            if health.get('reason') in {'rate_limit', 'transient'}:
                self.next_scan = max(self.clock() + 1, health.get('retry_at', self.next_scan))
            self.next_prices = now + self.owner.policy.price_refresh_interval
        elif now >= self.next_prices:
            self.refresh_prices()
            self.next_prices = now + self.owner.policy.price_refresh_interval
        health = self.snapshot.get('market_health', {})
        health['heartbeat'] = iso(self.clock())
        self.snapshot['market_health'] = health
        self.owner.publish_scan(self.snapshot)
        self.owner.process_pending()
        if self.automation is not None:
            self.automation.tick()
            self.owner.process_pending()

    def refresh_prices(self) -> None:
        coverage = self.snapshot.get('market_health', {}).get('coverage', [])
        if not coverage:
            return
        try:
            weight = 2 if len(coverage) <= 20 else 40 if len(coverage) <= 100 else 80
            tickers = self._request('reduction', weight, lambda: self.provider.tickers(coverage))
            ticker_received = self.clock()
            metadata = self.snapshot.get('market_metadata', {})
            by_symbol = {s['baseAsset'] + '/' + s['quoteAsset']: s for s in metadata.get('symbols', [])}
            filter_prices = {}
            for symbol in coverage:
                notional = [f for f in by_symbol.get(symbol, {}).get('filters', []) if f['filterType'] in {'MIN_NOTIONAL', 'NOTIONAL'}]
                if not notional:
                    continue
                filter_prices[symbol] = self._filter_price(symbol, notional, 'reduction')
            metadata['filter_prices'] = filter_prices
            self.owner.publish_prices({s: tickers[s]['last'] for s in coverage}, fetched_at=ticker_received)
            self.snapshot['market_health'].pop('price_refresh_failure', None)
        except (TimeoutError, ConnectionError, ValueError, KeyError, ProviderFailure) as error:
            # Preserve both value and fetched-at on failure; no cached-price publication.
            self.snapshot['market_health']['price_refresh_failure'] = error.kind if isinstance(error, ProviderFailure) else 'data_quality'
        self.owner.publish_scan(self.snapshot)

    def _filter_price(self, symbol: str, notional: list[dict[str, Any]], lane: str) -> dict[str, Any]:
        reference = self._request(lane, 2, lambda: self.provider.reference_price(symbol))
        observation = {'reference': reference, 'fetched_at': iso(self.clock())}
        if reference.get('referencePrice') is None and any(f.get('avgPriceMins', 0) for f in notional):
            observation['average'] = self._request(lane, 2, lambda: self.provider.average_price(symbol))
            observation['fetched_at'] = iso(self.clock())
        return observation

    def _candles(self, symbol: str, timeframe: str, server_time: float, *, required: int | None = None) -> list[list[float]]:
        duration, default_required = INTERVALS[timeframe]
        required = default_required if required is None else required
        boundary = int(server_time // duration) * duration
        key = f'{self.owner.store.venue}:{symbol}:{timeframe}'
        cached = self.cache.get(key, {})
        if not isinstance(cached, dict):
            raise DataQualityError('Invalid cached candles')
        reused = cached.get('closed_at') == boundary and len(cached.get('rows',[])) >= required
        lane = 'reduction' if symbol in self.owner.store.market_coverage() else 'market'
        incoming = cached['rows'] if reused else self._request(lane, 2, lambda: self.provider.candles(symbol, timeframe, required + 1))
        rows = validated_bars(incoming, timeframe, server_time,required=required)
        self.cache[key] = {'venue': self.owner.store.venue, 'symbol': symbol, 'timeframe': timeframe,
                           'closed_at': boundary, 'required':required, 'fetched_at': cached['fetched_at'] if reused else iso(self.clock()), 'rows': rows}
        return rows

    def scan_once(self) -> None:
        now = self.clock()
        version, config = self.owner.store.config()
        health = dict(self.snapshot.get('market_health', {}))
        if health.get('reason') == 'auth' or now < health.get('retry_at', 0):
            health['heartbeat'] = iso(now)
            self.snapshot['market_health'] = health
            self.owner.publish_scan(self.snapshot)
            return
        health.update(heartbeat=iso(now), config_version=version, owner_generation=self.owner.generation,
                      state='degraded', reason='data_quality')
        financial_coverage = self.owner.store.market_coverage()
        self.budget.reserve_weight = 101 + 10 * max(1, len(financial_coverage))
        self.budget.reserve_requests = 3 + 5 * max(1, len(financial_coverage))
        try:
            before = self.clock()
            metadata = self._request('reconciliation', 21, self.provider.metadata, raw_requests=2, discovery=True)
            received = self.clock()
            server_time = float(metadata['serverTime']) / 1000
            # Clock tolerance uses the selected price-expiry bound, not an invented trading policy.
            health['clock_skew_seconds'] = server_time - (before + received) / 2
            health['clock_rtt_seconds'] = received - before
            if not math.isfinite(server_time) or abs(health['clock_skew_seconds']) >= self.owner.policy.price_expiry or not 0 <= received - before < self.owner.policy.price_expiry:
                raise DataQualityError('Clock uncertainty')
            self.budget.configure(metadata, request_started=before)
            self.budget.observe(getattr(self.provider, 'headers', {}))
            tickers = self._request('reduction' if financial_coverage else 'market', 80, self.provider.tickers)
            ticker_received = self.clock()
            self.snapshot['market_metadata'] = {'venue': self.owner.store.venue, 'fetched_at': iso(received), 'symbols': metadata['symbols']}
            valid = {f"{s['baseAsset']}/{s['quoteAsset']}" for s in metadata['symbols']
                     if s.get('status') == 'TRADING' and s.get('isSpotTradingAllowed') is True}
            eligible = {s for s in valid if s.endswith('/' + config['base_quote']) and s not in self.blacklist
                        and (not self.whitelist or s in self.whitelist)
                        and not any(x in s for x in ('UP/', 'DOWN/', 'BULL/', 'BEAR/'))}
            ranked = sorted((s for s in eligible if
                             s in tickers and float(tickers[s].get('quoteVolume') or 0) >= config['min_quote_vol']),
                            key=lambda s: (-float(tickers[s]['quoteVolume']), s))
            fallback = [f'{base}/{config["base_quote"]}' for base in ('ETH', 'BTC', 'SOL', 'BNB')
                        if f'{base}/{config["base_quote"]}' in eligible]
            universe = (ranked if config['use_dynamic_universe'] and ranked else fallback)[:config['universe_size']]
            required = set(universe) | self.owner.store.market_coverage()
            health['coverage'] = sorted(required)
            if not required or not required <= valid:
                raise ProviderFailure('invalid_market')
            filter_prices = {}
            by_symbol = {s['baseAsset'] + '/' + s['quoteAsset']: s for s in metadata['symbols']}
            for symbol in sorted(required):
                notional = [f for f in by_symbol[symbol].get('filters', []) if f['filterType'] in {'MIN_NOTIONAL', 'NOTIONAL'}]
                if not notional:
                    continue
                filter_prices[symbol] = self._filter_price(symbol, notional, 'reduction' if symbol in financial_coverage else 'market')
            self.snapshot['market_metadata']['filter_prices'] = filter_prices
            from strategy_profiles import PROFILES
            observation = self.owner.store.snapshot(now=self.clock())
            profiles = {config['strategy_profile']} | {r['config']['strategy_profile'] for r in observation['position_rules'].values() if r['state'] == 'bound'}
            requirements: dict[str,int] = {}
            for name in profiles:
                contract = PROFILES[name]
                for timeframe, count in ((contract.trend,contract.trend_window),(contract.bias,contract.bias_window),('15m',contract.execution_window)):
                    requirements[timeframe] = max(requirements.get(timeframe,0),count)
            timeframes = set(requirements)
            health['timeframes'] = sorted(timeframes)
            for symbol in sorted(required, key=lambda s: (s not in financial_coverage, s)):
                for timeframe in sorted(timeframes):
                    self._candles(symbol, timeframe, server_time,required=requirements[timeframe])
            self.owner.publish_prices({s: tickers[s]['last'] for s in required}, fetched_at=ticker_received)
            if self.evaluate is not None:
                self.snapshot.update(self.evaluate(CachedMarketView(self.owner.store.venue, self.cache, tickers,context={
                    'namespace':self.owner.store.namespace,'config_version':version,
                    'as_of':datetime.fromtimestamp(server_time,timezone.utc),
                    'positions':observation['positions'],'position_rules':observation['position_rules']}), universe, config))
            health.update(state='ready', reason=None, last_completed_scan=iso(self.clock()), coverage=sorted(required), failures=0, retry_at=0)
            health.pop('price_refresh_failure', None)
            self.snapshot.update(universe=universe, last_scan=health['last_completed_scan'])
        except (DataQualityError, ValueError, KeyError, TypeError, IndexError):
            health.update(state='degraded', reason='data_quality')
        except (TimeoutError, ConnectionError):
            health.update(state='degraded', reason='transient')
            self._backoff(health, ProviderFailure('transient'), now)
        except BudgetDeferred as error:
            health.update(state='degraded', reason='rate_limit', retry_at=error.retry_at)
        except ProviderFailure as error:
            health.update(state='degraded', reason=error.kind)
            self._backoff(health, error, now)
        if health['state'] != 'ready':
            self.snapshot.update(signals=[], exit_signals=[])
        self.snapshot.update(market_health=health, market_cache=self.cache, config_version=version)
        self.owner.publish_scan(self.snapshot)

    def _backoff(self, health: dict[str, Any], error: ProviderFailure, now: float) -> None:
        failures = min(int(health.get('failures', 0)) + 1, 6)
        health.update(failures=failures, retry_at=now + max(error.retry_after, min(2 ** failures, 60)) + random.random())


def observed_health(snapshot: dict[str, Any], now: float, timeout: int = OperationsPolicy().scan_health_timeout) -> dict[str, Any]:
    health = dict(snapshot.get('market_health', {'state': 'degraded', 'reason': 'scanner_uninitialized'}))
    completed = health.get('last_completed_scan')
    try:
        if not isinstance(completed, str):
            raise ValueError('Missing completed scan')
        age = now - datetime.fromisoformat(completed).timestamp()
        health['scan_age_seconds'] = age
        if not 0 <= age < timeout:
            health.update(state='degraded', reason=health.get('reason') or 'scanner_stale')
    except (TypeError, ValueError, OverflowError):
        health.update(state='degraded', reason=health.get('reason') or 'scanner_uninitialized')
    health['execution_ready'] = health['state'] == 'ready'
    ownership = snapshot.get('ownership')
    if ownership is not None and ownership.get('state') != 'ready':
        health.update(execution_ready=False, state='degraded', reason=health.get('reason') or 'owner_unavailable')
    if snapshot.get('config_version') != health.get('config_version'):
        health.update(execution_ready=False, state='degraded', reason='configuration_changed')
    if health['execution_ready']:
        try:
            server_time = now + float(health['clock_skew_seconds'])
            for symbol in health.get('coverage', []):
                for timeframe in health.get('timeframes', ['4h','1h','15m']):
                    duration, _ = INTERVALS[timeframe]
                    key = f'{snapshot["market_prices"][symbol]["venue"]}:{symbol}:{timeframe}'
                    cached = snapshot['market_cache'][key]
                    if cached['closed_at'] != int(server_time // duration) * duration:
                        raise DataQualityError('Missing latest completed candle')
                    validated_bars(cached['rows'], timeframe, server_time,required=cached.get('required'))
        except (DataQualityError, KeyError, TypeError, ValueError, IndexError, OverflowError):
            health.update(execution_ready=False, state='degraded', reason='candle_stale')
    if health['execution_ready']:
        for symbol in health.get('coverage', []):
            try:
                quote = snapshot['market_prices'][symbol]
                price = Decimal(quote['price'])
                stamp = datetime.fromisoformat(quote['fetched_at'])
                if not price.is_finite() or price <= 0 or stamp.tzinfo is None or not 0 <= now - stamp.timestamp() < OperationsPolicy().price_expiry:
                    raise ValueError('Stale price')
            except (KeyError, ValueError, TypeError, InvalidOperation):
                health.update(execution_ready=False, state='degraded', reason='price_stale')
                break
    if health['execution_ready'] and health.get('price_refresh_failure'):
        health.update(execution_ready=False, state='degraded', reason=health['price_refresh_failure'])
    if health['execution_ready']:
        budget = snapshot.get('market_budget', {})
        if budget.get('state') != 'configured' or now < budget.get('blocked_until', 0):
            health.update(execution_ready=False, state='degraded', reason=budget.get('reason') or 'market_budget_unavailable')
    return health
