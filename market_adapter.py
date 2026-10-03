"""Explicit Binance Spot public REST adapter; no implicit market/private loads."""
from __future__ import annotations
from typing import Any
import json
import math
from request_budget import ProviderFailure


class BinancePublicMarket:
    def __init__(self, exchange: Any) -> None:
        self.exchange = exchange
        self.exchange.timeout = 10000
        self.exchange.maxRetriesOnFailure = 0
        self.symbols: dict[str, str] = {}

    @property
    def headers(self) -> dict[str, Any]:
        return dict(self.exchange.last_response_headers or {})

    def _get(self, method: str, params: dict[str, Any] | None = None) -> Any:
        try:
            self.exchange.last_response_headers = None
            if method == 'public_get_referenceprice' and not hasattr(self.exchange, method):
                return self.exchange.request('referencePrice', 'public', 'GET', params or {})
            return getattr(self.exchange, method)(params or {})
        except Exception as error:
            if method == 'public_get_referenceprice' and '-2043' in str(error):
                return {'referencePrice': None}
            name = type(error).__name__
            kind = 'data_quality'
            if name in {'AuthenticationError', 'PermissionDenied', 'AccountSuspended'}:
                kind = 'auth'
            elif name in {'RateLimitExceeded', 'DDoSProtection'}:
                kind = 'rate_limit'
            elif name in {'TimeoutError', 'ConnectionError', 'NetworkError', 'RequestTimeout', 'ExchangeNotAvailable', 'OnMaintenance'}:
                kind = 'transient'
            elif name == 'BadSymbol':
                kind = 'invalid_market'
            try:
                retry = float(next((v for k, v in self.headers.items() if k.lower() == 'retry-after'), 0))
                retry = retry if math.isfinite(retry) and retry >= 0 else 0
            except (TypeError, ValueError):
                retry = 0
            raise ProviderFailure(kind, retry_after=retry) from None

    def metadata(self) -> dict[str, Any]:
        metadata = dict(self._get('public_get_exchangeinfo'))
        metadata['serverTime'] = self._get('public_get_time')['serverTime']
        self.symbols = {f"{s['baseAsset']}/{s['quoteAsset']}": s['symbol'] for s in metadata['symbols']}
        return metadata

    def tickers(self, symbols: list[str] | None = None) -> dict[str, Any]:
        params = {} if symbols is None else {'symbols': json.dumps([self.symbols[s] for s in symbols])}
        rows = self._get('public_get_ticker_24hr', params)
        names = {raw: normalized for normalized, raw in self.symbols.items()}
        return {names[row['symbol']]: {'last': float(row['lastPrice']), 'quoteVolume': float(row['quoteVolume'])}
                for row in rows if row['symbol'] in names}

    def reference_price(self, symbol: str) -> dict[str, Any]:
        return dict(self._get('public_get_referenceprice', {'symbol': self.symbols[symbol]}))

    def average_price(self, symbol: str) -> dict[str, Any]:
        return dict(self._get('public_get_avgprice', {'symbol': self.symbols[symbol]}))

    def candles(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        rows = self._get('public_get_klines', {'symbol': self.symbols[symbol], 'interval': timeframe, 'limit': limit})
        return [[float(value) for value in row[:6]] for row in rows]

    def candles_range(self, symbol: str, timeframe: str, start: int, end: int,
                      limit: int) -> list[list[float]]:
        from strategy_profiles import DURATIONS
        if (timeframe not in DURATIONS or type(start) is not int or type(end) is not int
                or not 0 <= start < end or type(limit) is not int or not 1 <= limit <= 1000):
            raise ValueError('Explicit UTC range and bounded page required')
        rows=self._get('public_get_klines', {'symbol':self.symbols[symbol],'interval':timeframe,
            'startTime':start,'endTime':end-1,'limit':limit,'timeZone':'0'})
        return [[float(value) for value in row[:6]] for row in rows]
