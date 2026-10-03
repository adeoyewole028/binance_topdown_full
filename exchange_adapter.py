"""Pinned CCXT precision with explicit Spot-only, individually budgeted REST."""
from __future__ import annotations

import json
import math
from typing import Any
from urllib.parse import urlparse
from request_budget import ProviderFailure
from runtime_environment import RuntimeEnvironment, construct_exchange


class SubmissionRejected(ProviderFailure):
    """Only an authoritative pre-engine rejection can release unknown funds."""


def error_payload(error: Exception) -> dict[str, Any]:
    text = str(error)
    start = text.find('{')
    if start < 0:
        return {}
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
        return value if isinstance(value, dict) else {}
    except ValueError:
        return {}


class BinanceSpotOrders:
    def __init__(self, runtime: RuntimeEnvironment, *, exchange: Any = None) -> None:
        runtime.require_exchange()
        if runtime.mode != 'exchange':
            raise ValueError('Private Spot adapter requires exchange execution mode')
        self.venue = runtime.venue
        self.exchange = construct_exchange(runtime, private=True) if exchange is None else exchange
        self.exchange.timeout = 10000
        self.exchange.maxRetriesOnFailure = 0
        self.exchange.options['maxRetriesOnFailure'] = 0
        self._verify_venue()

    def _verify_venue(self) -> None:
        host = urlparse(self.exchange.urls['api']['private']).hostname
        allowed = {'testnet.binance.vision'} if self.venue == 'testnet' else {'api.binance.com', 'api-gcp.binance.com', 'api1.binance.com', 'api2.binance.com', 'api3.binance.com', 'api4.binance.com'}
        if host not in allowed:
            raise ProviderFailure('data_quality')

    @property
    def headers(self) -> dict[str, Any]:
        return dict(self.exchange.last_response_headers or {})

    def configure(self, metadata: dict[str, Any]) -> None:
        self._verify_venue()
        if metadata.get('venue') != self.venue or not metadata.get('symbols'):
            raise ProviderFailure('data_quality')
        # Parsing and installation are local. Unified private helpers that can
        # silently load markets or batch requests are deliberately not invoked.
        self.exchange.options['crossMarginPairsData'] = []
        self.exchange.options['isolatedMarginPairsData'] = []
        markets = [self.exchange.parse_market(row) for row in metadata['symbols']]
        self.exchange.set_markets(markets)

    def quantity(self, symbol: str, value: str) -> str:
        return str(self.exchange.amount_to_precision(symbol, value))

    def _symbol(self, symbol: str) -> str:
        return str(self.exchange.market(symbol)['id'])

    def _call(self, method: str, params: dict[str, Any]) -> Any:
        self._verify_venue()
        self.exchange.last_response_headers = None
        try:
            return getattr(self.exchange, method)(params)
        except Exception as error:
            payload = error_payload(error)
            code = payload.get('code')
            code = code if type(code) is int else None
            if method in {'private_get_order', 'private_get_orderlist'} and type(code) is int and code == -2013:
                return None  # Never evidence that an uncertain submit failed.
            name = type(error).__name__
            kind = 'auth' if code in {-2014, -2015, -1022} or name in {'AuthenticationError', 'PermissionDenied'} else 'rate_limit' if name in {'RateLimitExceeded', 'DDoSProtection'} else 'transient' if name in {'TimeoutError', 'ConnectionError', 'NetworkError', 'RequestTimeout', 'ExchangeNotAvailable'} else 'data_quality'
            try:
                retry = float(next((v for k, v in self.headers.items() if k.lower() == 'retry-after'), 0))
                retry = retry if math.isfinite(retry) and retry >= 0 else 0
            except (TypeError, ValueError):
                retry = 0
            failure = SubmissionRejected if method in {'private_post_order', 'private_post_orderlist_oco'} and type(code) is int and code in {-1013, -1021, -1022, -2014, -2015} else ProviderFailure
            raise failure(kind, retry_after=retry) from None

    def account(self) -> dict[str, Any]:
        return dict(self._call('private_get_account', {'omitZeroBalances': False}))

    def submit(self, symbol: str, side: str, quantity: str, client_id: str) -> dict[str, Any]:
        result = dict(self._call('private_post_order', {'symbol': self._symbol(symbol), 'type': 'MARKET',
                                                    'side': side.upper(), 'quantity': quantity,
                                                    'newClientOrderId': client_id, 'newOrderRespType': 'FULL'}))
        stamp = result.get('transactTime')
        if type(stamp) is not int or stamp < 0:
            raise ProviderFailure('data_quality')
        result['updateTime'] = stamp
        return result

    def query(self, symbol: str, client_id: str) -> dict[str, Any] | None:
        value = self._call('private_get_order', {'symbol': self._symbol(symbol), 'origClientOrderId': client_id})
        return dict(value) if value is not None else None

    def query_order(self, symbol: str, order_id: str) -> dict[str, Any] | None:
        value = self._call('private_get_order', {'symbol': self._symbol(symbol), 'orderId': order_id})
        return dict(value) if value is not None else None

    def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return list(self._call('private_get_openorders', {'symbol': self._symbol(symbol)} if symbol else {}))

    def trades(self, symbol: str, *, from_id: int = 0, order_id: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {'symbol': self._symbol(symbol), 'fromId': from_id, 'limit': 1000}
        if order_id is not None:
            params['orderId'] = order_id
        return list(self._call('private_get_mytrades', params))

    def protect(self, symbol: str, quantity: str, stop: str, target: str,
                list_id: str, above_id: str, below_id: str) -> dict[str, Any]:
        return dict(self._call('private_post_orderlist_oco', {
            'symbol': self._symbol(symbol), 'side': 'SELL', 'quantity': quantity,
            'listClientOrderId': list_id, 'aboveType': 'TAKE_PROFIT', 'aboveClientOrderId': above_id,
            'aboveStopPrice': target, 'belowType': 'STOP_LOSS', 'belowClientOrderId': below_id,
            'belowStopPrice': stop, 'newOrderRespType': 'FULL'}))

    def query_list(self, list_id: str) -> dict[str, Any] | None:
        result = self._call('private_get_orderlist', {'origClientOrderId': list_id})
        return dict(result) if result is not None else None

    def cancel_list(self, symbol: str, list_id: str, cancel_id: str) -> dict[str, Any]:
        return dict(self._call('private_delete_orderlist', {'symbol': self._symbol(symbol),
                                                         'listClientOrderId': list_id, 'newClientOrderId': cancel_id}))
