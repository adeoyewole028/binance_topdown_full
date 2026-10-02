"""Validated launch and provider boundary; credentials always fictitious."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from unittest.mock import patch
import runpy
from types import SimpleNamespace

import pytest

from tests.legacy_sandbox import LegacySandbox, ROOT


@pytest.mark.parametrize('env', [
    {'TRADING_MODE': 'exchange'},
    {'TRADING_MODE': 'paper', 'PAPER_TRADING': 'false'},
    {'BINANCE_VENUE': 'mainnet', 'TESTNET': 'true'},
    {'BINANCE_VENUE': 'futures'},
    {'ALLOW_EXCHANGE_EXECUTION': 'perhaps'},
    {'API_KEY': 'wrong-venue', 'API_SECRET': 'wrong-secret'},
    {'API_KEY': 'a', 'BINANCE_API_KEY': 'b', 'API_SECRET': 'c', 'CREDENTIAL_VENUE': 'testnet'},
    {'FLASK_DEBUG': 'true'},
    {'FLASK_RUN_RELOAD': 'true'},
    {'GUNICORN_CMD_ARGS': '--reload'},
    {'TRADING_MODE': 'exchange', 'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true',
     'BINANCE_VENUE': 'mainnet', 'BINANCE_MAINNET_API_KEY': 'offline-key', 'BINANCE_MAINNET_API_SECRET': 'offline-secret'},
])
def test_invalid_environment_fails_before_private_calls(env: dict[str, str]) -> None:
    with LegacySandbox() as sandbox:
        with pytest.raises(ValueError):
            sandbox.module.initialize_runtime({**os.environ, **env})
        assert sandbox.exchange.orders == []


@pytest.mark.parametrize('venue', ['testnet', 'mainnet'])
def test_daemon_and_dashboard_share_venue_and_public_credentials(venue: str) -> None:
    with LegacySandbox(environment={'BINANCE_VENUE': venue}) as sandbox:
        calls: list[tuple[str, Any]] = []
        class ObservedVenue:
            def set_sandbox_mode(self, enabled: bool) -> None:
                calls.append(('sandbox', enabled))
        def constructor(config: dict[str, Any]) -> ObservedVenue:
            calls.append(('construct', config))
            return ObservedVenue()
        with patch('ccxt.binance', constructor):
            daemon = sandbox.load_module('topdown_daemon')
            daemon.make_exchange()
        assert calls[0][0] == 'construct'
        config = calls[0][1]
        assert config['options']['defaultType'] == 'spot'
        assert 'apiKey' not in config and 'secret' not in config
        assert calls[1:] == ([('sandbox', True)] if venue == 'testnet' else [])


def test_normal_imports_in_fresh_process_have_no_startup_effects(tmp_path: Path) -> None:
    result = subprocess.run([sys.executable, '-c', '''
import importlib
from tests.legacy_sandbox import LegacySandbox
with LegacySandbox() as sandbox:
    for name in ('dashboard', 'topdown_daemon', 'backtester'):
        importlib.import_module(name)
    assert sandbox.thread_start_attempts == 0
    assert sandbox.network_attempts == 0
    assert sandbox.forbidden_file_attempts == 0
    assert sandbox.exchange.orders == []
    assert sandbox.removed_startup == []
print('Safe normal imports')
'''], cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'Safe normal imports'


def test_explicit_wsgi_launch_is_http_only_and_requires_production_secrets() -> None:
    result = subprocess.run([sys.executable, '-c', '''
import os, runpy
from unittest.mock import patch
from tests.legacy_sandbox import LegacySandbox
with LegacySandbox(environment={'APP_ENV': 'production'}, authenticated=False) as sandbox:
    with patch('dotenv.load_dotenv', lambda *a, **kw: None):
        namespace = runpy.run_path('wsgi.py')
        app = namespace['app']
        assert app.test_client().get('/login').status_code == 200
        assert app.test_client().get('/api/status').status_code == 401
        assert sandbox.thread_start_attempts == 0
        assert sandbox.network_attempts == 0
        with patch.dict(os.environ, {'FLASK_SECRET_KEY': 'supersecretkey123'}):
            try:
                runpy.run_path('wsgi.py')
            except ValueError:
                pass
            else:
                raise AssertionError('Default secret was accepted')
print('Safe production launch')
'''], cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'Safe production launch'


def test_private_testnet_launch_applies_sandbox_before_provider_methods() -> None:
    with LegacySandbox(environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        calls: list[tuple[str, Any]] = []
        class Venue:
            def set_sandbox_mode(self, enabled: bool) -> None:
                calls.append(('sandbox', enabled))
        def constructor(config: dict[str, Any]) -> Venue:
            calls.append(('construct', config))
            return Venue()
        with patch('ccxt.binance', constructor):
            sandbox.load_module('topdown_daemon').make_exchange()
        assert calls[0][1]['apiKey'] == 'offline-key'
        assert calls[0][1]['secret'] == 'offline-secret'
        assert calls[1:] == [('sandbox', True)]


def test_persisted_exchange_flag_cannot_override_default_paper_startup(tmp_path: Path) -> None:
    (tmp_path / 'config.json').write_text('{"paper_trading": false}', encoding='utf-8')
    with LegacySandbox(tmp_path) as sandbox:
        assert sandbox.client.get('/api/status').get_json()['execution_mode'] == 'paper'
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'})
        assert sandbox.exchange.holdings == {}


def test_explicit_development_launcher_binds_loopback_without_debug_or_scanner() -> None:
    result = subprocess.run([sys.executable, '-c', '''
import runpy, sys
from unittest.mock import patch
from tests.legacy_sandbox import LegacySandbox
with LegacySandbox(authenticated=False) as sandbox:
    started = []
    def server(app, **settings):
        started.append(settings)
        assert app.test_client().get('/login').status_code == 200
    with patch('dotenv.load_dotenv', lambda *a, **kw: None), patch('flask.Flask.run', server), patch.object(sys, 'argv', ['dashboard.py']):
        runpy.run_path('dashboard.py', run_name='__main__')
    assert started == [{'host': '127.0.0.1', 'port': 5000, 'debug': False, 'use_reloader': False}]
    assert sandbox.thread_start_attempts == 0
    assert sandbox.network_attempts == 0
print('Safe development launch')
'''], cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'Safe development launch'


def test_production_server_rejects_effective_reload_configuration() -> None:
    # The server hook is Gunicorn's public launcher boundary; the host is replaced
    # on Windows so no unsupported Unix master process or listener is started.
    launcher = runpy.run_path(str(ROOT / 'gunicorn.conf.py'))
    for reload in (False, True):
        server = SimpleNamespace(cfg=SimpleNamespace(reload=reload))
        if reload:
            with pytest.raises(ValueError, match='reload'):
                launcher['on_starting'](server)
            with pytest.raises(ValueError, match='reload'):
                launcher['on_reload'](server)
        else:
            launcher['on_starting'](server)
