"""Disposable operator/evaluator boundary with real, unmodified normal imports.

Transport/credential/write guards stay active; only fake keys and temp state exist.
BT02 removed the AST startup workaround originally recorded in BT01 evidence.
"""
from __future__ import annotations

import builtins
from contextlib import ExitStack
import io
import os
from pathlib import Path
import sys
import tempfile
import re
import time
from types import ModuleType, TracebackType
from typing import Any, Literal, Self
from unittest.mock import patch

from flask.testing import FlaskClient
import pandas as pd  # Load dependency before temporarily blocking thread startup.
from werkzeug.security import generate_password_hash

ROOT = Path(__file__).resolve().parents[1]
OFFLINE_PASSWORD_HASH = generate_password_hash('offline-password')


class OfflineViolation(RuntimeError):
    """The test tried to leave its disposable, credential-free boundary."""


class FakeExchange:
    """External Spot boundary: observable simulated exchange holdings, no IO."""

    def __init__(self) -> None:
        self.holdings: dict[str, float] = {}
        self.prices: dict[str, float] = {}
        self.market_filters: dict[str, list[dict[str, Any]]] = {}
        self.orders: list[dict[str, Any]] = []
        self.sandbox = False

    def set_sandbox_mode(self, enabled: bool) -> None:
        self.sandbox = enabled

    def create_order(
        self, symbol: str, order_type: str, side: str, qty: float,
        price: float | None = None, params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        execution_price = self.prices.get(symbol, 100.0) if price is None else price
        self.holdings[symbol] = self.holdings.get(symbol, 0) + (qty if side == 'buy' else -qty)
        order = {
            'id': f'offline-{len(self.orders) + 1}', 'symbol': symbol,
            'type': order_type, 'side': side, 'amount': qty, 'filled': qty,
            'price': execution_price, 'average': execution_price, 'status': 'closed',
        }
        self.orders.append(order)
        return order

    def fetch_ticker(self, symbol: str) -> dict[str, float]:
        return {'last': self.prices.get(symbol, 100.0)}

    def metadata(self) -> dict[str, Any]:
        return {'serverTime': int(time.time() * 1000), 'rateLimits': [
            {'rateLimitType': 'REQUEST_WEIGHT', 'interval': 'MINUTE', 'intervalNum': 1, 'limit': 100000}],
            'symbols': [{'symbol': s.replace('/', ''), 'baseAsset': s.split('/')[0], 'quoteAsset': s.split('/')[1],
                         'status': 'TRADING', 'isSpotTradingAllowed': True, 'ocoAllowed': True,
                         'orderTypes': ['MARKET', 'STOP_LOSS', 'TAKE_PROFIT'],
                         'filters': self.market_filters.get(s, [{'filterType': 'LOT_SIZE', 'minQty': '0.001', 'maxQty': '100000', 'stepSize': '0.001'},
                                     {'filterType': 'MIN_NOTIONAL', 'minNotional': '5', 'applyToMarket': True, 'avgPriceMins': 0}]) +
                                    [{'filterType': 'PRICE_FILTER', 'minPrice': '0.01', 'maxPrice': '1000000', 'tickSize': '0.01'}]} for s in self.prices]}

    def tickers(self, symbols: list[str] | None = None) -> dict[str, Any]:
        return {s: {'last': p, 'quoteVolume': 20000000.0} for s, p in self.prices.items()
                if symbols is None or s in symbols}

    def reference_price(self, symbol: str) -> dict[str, Any]:
        return {'referencePrice': None}

    def average_price(self, symbol: str) -> dict[str, Any]:
        return {'mins': 5, 'price': str(self.prices[symbol])}

    def candles(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        duration = {'4h': 14400, '1h': 3600, '15m': 900}[timeframe]
        boundary = int(time.time() // duration) * duration
        price = self.prices[symbol]
        return [[float((boundary - (limit - index) * duration) * 1000), price, price + 1,
                 max(price - 1, price / 2), price, 10.0] for index in range(limit)]


class LegacySandbox:
    def __init__(self, data_directory: Path | None = None, *, authenticated: bool = True,
                 environment: dict[str, str] | None = None, paper_setup: bool = True) -> None:
        self.stack = ExitStack()
        self.provided_directory = data_directory
        self.exchange = FakeExchange()
        self.network_attempts = 0
        self.thread_start_attempts = 0
        self.forbidden_file_attempts = 0
        self.removed_startup: list[str] = []
        self.real_open = builtins.open
        self.real_io_open = io.open
        self.real_mkdir = os.mkdir
        self.module = ModuleType('not_loaded')
        self.client: FlaskClient
        self.directory: Path
        self.authenticated = authenticated
        self.environment = environment or {}
        self.csrf = ''
        self.paper_setup = paper_setup

    def __enter__(self) -> Self:
        try:
            self.directory = self.provided_directory or Path(
                self.stack.enter_context(tempfile.TemporaryDirectory(prefix='bt01-')),
            )
            self.directory.mkdir(parents=True, exist_ok=True)
            self.stack.enter_context(patch.dict(os.environ, {
                'DATA_DIR': str(self.directory), 'PAPER_TRADING': 'true',
                'APP_ENV': 'development',
                'FLASK_SECRET_KEY': 'offline-session-secret-with-32-characters',
                'OPERATOR_USERNAME': 'operator', 'OPERATOR_PASSWORD_HASH': OFFLINE_PASSWORD_HASH,
                'BINANCE_TESTNET_API_KEY': 'offline-key', 'BINANCE_TESTNET_API_SECRET': 'offline-secret',
                **self.environment,
            }, clear=True))
            self.stack.enter_context(patch.object(sys, 'dont_write_bytecode', True))
            fake_ccxt = ModuleType('ccxt')
            setattr(fake_ccxt, 'binance', lambda *args, **kwargs: self.exchange)
            fake_dotenv = ModuleType('dotenv')
            setattr(fake_dotenv, 'load_dotenv', self._forbid_credentials)
            self.stack.enter_context(patch.dict(sys.modules, {
                'ccxt': fake_ccxt, 'dotenv': fake_dotenv,
            }))
            for target in (
                'socket.socket.connect', 'socket.socket.connect_ex',
                'socket.create_connection', 'socket.getaddrinfo', 'socket.socket.sendto',
            ):
                self.stack.enter_context(patch(target, self._forbid_network))
            self.stack.enter_context(patch('threading.Thread.start', self._forbid_thread))
            self.stack.enter_context(patch('builtins.open', self._guarded_open))
            self.stack.enter_context(patch('io.open', self._guarded_io_open))
            self.stack.enter_context(patch('os.mkdir', self._guarded_mkdir))
            self.module = self.load_module('dashboard')
            from state_store import StateStore
            from runtime_environment import RuntimeEnvironment
            runtime = RuntimeEnvironment.from_mapping(os.environ)
            store = StateStore(runtime.data_directory, os.getenv('LOCAL_ACCOUNT_ID', 'operator-account'), runtime.venue, runtime.mode)
            store.provision(self.module.environment_defaults(os.environ))
            if self.paper_setup and store.mode == 'paper' and not store.config()[1].get('paper_policy'):
                store.setup_paper({'quote': 'USDT', 'initial_cash': '10000', 'fee_fraction': '0', 'slippage_fraction': '0',
                                   'risk_fraction': '1', 'fixed_quote': '100', 'max_allocation_fraction': '1',
                                   'max_exposure_fraction': '1', 'max_positions': '100', 'cooldown_seconds': '0',
                                   'drawdown_fraction': '1'}, 'explicit-offline-fixture', expected_version=store.config()[0])
            # Validated public-provider fixtures now accompany ordinary paper
            # orders. This keeps prior acceptance exercising fills, not safety bypasses.
            from execution_worker import ExecutionWorker
            from market_runtime import MarketRuntime
            enter_owner = ExecutionWorker.__enter__
            provider = self.exchange
            def owned_provider(worker: ExecutionWorker) -> ExecutionWorker:
                entered = enter_owner(worker)
                if provider.prices:
                    MarketRuntime(entered, provider, clock=worker.clock).scan_once()
                return entered
            self.stack.enter_context(patch.object(ExecutionWorker, '__enter__', owned_provider))
            self.module.initialize_runtime()
            self.module.__dict__['exchange'] = self.exchange
            self.module.app.config['TESTING'] = True
            class ReviewedClient(FlaskClient):
                def post(self, *args: Any, **kwargs: Any) -> Any:
                    # Ordinary fixture actions model the status review preceding
                    # submit; explicit empty/stale tokens remain untouched.
                    if args and args[0] in {'/trade', '/close'} and isinstance(kwargs.get('data'), dict):
                        data = dict(kwargs['data'])
                        if 'order_review' not in data:
                            observed = self.get('/api/status').get_json(silent=True) or {}
                            if 'order_review' in observed:
                                data['order_review'] = observed['order_review']
                        kwargs['data'] = data
                    return super().post(*args, **kwargs)
            self.module.app.test_client_class = ReviewedClient
            self.client = self.module.app.test_client()
            if self.authenticated:
                self.login()
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(
        self, error_type: type[BaseException] | None,
        error: BaseException | None, traceback: TracebackType | None,
    ) -> None:
        self.stack.close()

    def _forbid_network(self, *args: Any, **kwargs: Any) -> Any:
        self.network_attempts += 1
        raise OfflineViolation('Network access is forbidden in BT01 tests')

    def _forbid_thread(self, *args: Any, **kwargs: Any) -> Any:
        self.thread_start_attempts += 1
        raise OfflineViolation('Background startup is forbidden in BT01 tests')

    def _forbid_credentials(self, *args: Any, **kwargs: Any) -> Any:
        raise OfflineViolation('Loading a real .env is forbidden in BT01 tests')

    def _check_file(self, file: Any, mode: str) -> None:
        if isinstance(file, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(file)).resolve()
            if path.name == '.env' or (
                any(flag in mode for flag in 'wax+') and not path.is_relative_to(self.directory.resolve())
            ):
                self.forbidden_file_attempts += 1
                raise OfflineViolation('Credential reads and non-disposable writes are forbidden')

    def _guarded_open(self, file: Any, mode: str = 'r', *args: Any, **kwargs: Any) -> Any:
        self._check_file(file, mode)
        return self.real_open(file, mode, *args, **kwargs)

    def _guarded_io_open(self, file: Any, mode: str = 'r', *args: Any, **kwargs: Any) -> Any:
        self._check_file(file, mode)
        return self.real_io_open(file, mode, *args, **kwargs)

    def _guarded_mkdir(self, path: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:
        self._check_file(path, 'w')
        self.real_mkdir(path, mode, dir_fd=dir_fd)

    def load_module(
        self, name: Literal['dashboard', 'topdown_daemon', 'backtester', 'utils'],
        *, normal_import: bool = False,
    ) -> ModuleType:
        path = ROOT / f'{name}.py'
        source = path.read_text(encoding='utf-8')
        module = ModuleType(f'bt01_{name}')
        module.__file__ = str(path)
        self.stack.enter_context(patch.dict(sys.modules, {module.__name__: module}))
        exec(compile(source, str(path), 'exec'), module.__dict__)
        return module

    def login(self) -> None:
        page = self.client.get('/login').get_data(as_text=True)
        token = re.search(r'name="csrf_token" value="([^"]+)"', page)
        assert token is not None
        response = self.client.post('/login', data={
            'username': 'operator', 'password': 'offline-password', 'csrf_token': token[1],
        })
        assert response.status_code == 302
        page = self.client.get('/').get_data(as_text=True)
        token = re.search(r'name="csrf_token" value="([^"]+)"', page)
        assert token is not None
        self.csrf = token[1]

    def post(self, route: str, *, data: dict[str, str]) -> Any:
        response = self.client.post(route, data={**data, 'csrf_token': self.csrf})
        if route in {'/trade', '/close'} and response.status_code == 302:
            self.run_owner()
        return response

    def run_owner(self) -> None:
        """Explicit fake-clock consumer step at the approved end-to-end seam."""
        from execution_worker import ExecutionWorker
        with ExecutionWorker(self.module.state_store) as worker:
            worker.process_pending()

    def seed_price(self, symbol: str, price: float) -> None:
        """Legacy fixture provisioning; assertions use HTTP/provider boundaries."""
        self.module.status['live_prices'][symbol] = price
        self.exchange.prices[symbol] = price
        self.module.state_store.set_price(symbol, str(price))

    def candles(self, closes: list[float], timeframe_minutes: int = 15) -> pd.DataFrame:
        """UTC fixture bars are all completed before the fixture's decision time."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=len(closes), freq=f'{timeframe_minutes}min', tz='UTC'),
            'open': closes, 'high': [value + 1 for value in closes],
            'low': [value - 1 for value in closes], 'close': closes,
            'volume': [100.0] * len(closes),
        })
