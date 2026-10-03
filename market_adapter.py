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
            return getattr(self.exchange, method)(params or {})
        except Exception as error:
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

    def candles(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        rows = self._get('public_get_klines', {'symbol': self.symbols[symbol], 'interval': timeframe, 'limit': limit})
        return [[float(value) for value in row[:6]] for row in rows]
