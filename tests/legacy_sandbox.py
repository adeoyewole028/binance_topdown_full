"""Temporary BT01 adapter for the legacy HTTP and evaluator test seams.

Only legacy startup branches and daemon signal registration are stripped, as in
the original audit. This is characterization, not proof of safe normal imports.
The transport/credential/write guards remain active even for normal-import tests.
"""
from __future__ import annotations

import ast
import builtins
from contextlib import ExitStack
import io
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, TracebackType
from typing import Any, Literal, Self
from unittest.mock import patch

from flask.testing import FlaskClient
import pandas as pd  # Load dependency before temporarily blocking thread startup.

ROOT = Path(__file__).resolve().parents[1]


class OfflineViolation(RuntimeError):
    """The test tried to leave its disposable, credential-free boundary."""


class FakeExchange:
    """External Spot boundary: observable simulated exchange holdings, no IO."""

    def __init__(self) -> None:
        self.holdings: dict[str, float] = {}
        self.prices: dict[str, float] = {}
        self.orders: list[dict[str, Any]] = []

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


class LegacySandbox:
    def __init__(self, data_directory: Path | None = None) -> None:
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

    def __enter__(self) -> Self:
        try:
            self.directory = self.provided_directory or Path(
                self.stack.enter_context(tempfile.TemporaryDirectory(prefix='bt01-')),
            )
            self.directory.mkdir(parents=True, exist_ok=True)
            self.stack.enter_context(patch.dict(os.environ, {
                'DATA_DIR': str(self.directory), 'PAPER_TRADING': 'true',
                'FLASK_SECRET_KEY': 'offline-session-key',
                'API_KEY': 'offline-key', 'API_SECRET': 'offline-secret',
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
            self.module.local_broker.trade_log_path = str(self.directory / 'trades.csv')
            self.module.__dict__['exchange'] = self.exchange
            self.module.app.config['TESTING'] = True
            self.client = self.module.app.test_client()
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
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        if not normal_import and name != 'utils':
            tail = tree.body[-1]
            if not isinstance(tail, ast.If) or ast.unparse(tail.test) != "__name__ == '__main__'":
                raise OfflineViolation('Legacy startup changed; re-review this temporary adapter')
            tree.body.pop()
            self.removed_startup.append(f'{name}: terminal startup branch')
        if name == 'topdown_daemon':
            tree.body = [node for node in tree.body if not (
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == 'signal.signal'
            )]
            self.removed_startup.append('topdown_daemon: signal handler registration')
        module = ModuleType(f'bt01_{name}')
        module.__file__ = str(path)
        self.stack.enter_context(patch.dict(sys.modules, {module.__name__: module}))
        exec(compile(tree, str(path), 'exec'), module.__dict__)
        return module

    def seed_price(self, symbol: str, price: float) -> None:
        """Legacy fixture provisioning; assertions use HTTP/provider boundaries."""
        self.module.status['live_prices'][symbol] = price
        self.exchange.prices[symbol] = price

    def candles(self, closes: list[float], timeframe_minutes: int = 15) -> pd.DataFrame:
        """UTC fixture bars are all completed before the fixture's decision time."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=len(closes), freq=f'{timeframe_minutes}min', tz='UTC'),
            'open': closes, 'high': [value + 1 for value in closes],
            'low': [value - 1 for value in closes], 'close': closes,
            'volume': [100.0] * len(closes),
        })
