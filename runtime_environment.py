"""Pure environment validation and a single explicit Binance construction path."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys
import shlex
from typing import Any, Mapping


def parse_boolean(value: str) -> bool:
    value = value.strip().lower()
    if value in {'true', '1', 'yes', 'y', 'on'}:
        return True
    if value in {'false', '0', 'no', 'n', 'off'}:
        return False
    raise ValueError('Invalid boolean configuration')


def compatible_value(env: Mapping[str, str], canonical: str, aliases: tuple[str, ...],
                     default: str = '') -> str:
    values = [env[name] for name in (canonical, *aliases) if env.get(name)]
    if len(set(values)) > 1:
        raise ValueError(f'Conflicting {canonical} compatibility settings')
    return values[0] if values else default


@dataclass(frozen=True, repr=False)
class RuntimeEnvironment:
    development: bool
    mode: str
    venue: str
    allow_exchange: bool
    allow_mainnet: bool
    api_key: str
    api_secret: str
    data_directory: Path

    @classmethod
    def from_mapping(cls, env: Mapping[str, str]) -> RuntimeEnvironment:
        environment = compatible_value(env, 'APP_ENV', ('FLASK_ENV',), 'production')
        if environment not in {'development', 'production'}:
            raise ValueError('APP_ENV must be development or production')
        server_arguments = [*sys.argv, *shlex.split(env.get('GUNICORN_CMD_ARGS', ''))]
        if (any(parse_boolean(env.get(name, 'false')) for name in ('FLASK_DEBUG', 'FLASK_RUN_DEBUG', 'FLASK_RUN_RELOAD'))
                or any(argument.startswith('--reload') for argument in server_arguments)):
            raise ValueError('Debug and reloader startup are disabled')
        mode = env.get('TRADING_MODE', '') or ('paper' if parse_boolean(env.get('PAPER_TRADING', 'true')) else 'exchange')
        if mode not in {'paper', 'exchange'}:
            raise ValueError('TRADING_MODE must be paper or exchange')
        if env.get('PAPER_TRADING') and (mode == 'paper') != parse_boolean(env['PAPER_TRADING']):
            raise ValueError('Conflicting execution mode settings')
        venue = env.get('BINANCE_VENUE', '') or ('testnet' if parse_boolean(env.get('TESTNET', 'true')) else 'mainnet')
        if venue not in {'testnet', 'mainnet'}:
            raise ValueError('BINANCE_VENUE must be testnet or mainnet')
        if env.get('TESTNET') and (venue == 'testnet') != parse_boolean(env['TESTNET']):
            raise ValueError('Conflicting venue settings')
        key = compatible_value(env, f'BINANCE_{venue.upper()}_API_KEY', ('API_KEY', 'BINANCE_API_KEY'))
        secret = compatible_value(env, f'BINANCE_{venue.upper()}_API_SECRET', ('API_SECRET', 'BINANCE_API_SECRET'))
        if bool(key) != bool(secret):
            raise ValueError('Both venue credentials must be configured together')
        if any(env.get(name) for name in ('API_KEY', 'API_SECRET', 'BINANCE_API_KEY', 'BINANCE_API_SECRET')):
            if env.get('CREDENTIAL_VENUE') != venue:
                raise ValueError('Legacy credentials require matching CREDENTIAL_VENUE')
        if env.get('CREDENTIAL_VENUE') and env['CREDENTIAL_VENUE'] != venue:
            raise ValueError('Credential venue does not match configured venue')
        runtime = cls(environment == 'development', mode, venue,
                      parse_boolean(env.get('ALLOW_EXCHANGE_EXECUTION', 'false')),
                      parse_boolean(env.get('ALLOW_MAINNET_EXECUTION', 'false')),
                      key, secret, Path(env.get('DATA_DIR', str(Path(__file__).parent / 'data'))).resolve())
        if mode == 'exchange':
            runtime.require_exchange()
        return runtime

    def require_exchange(self) -> None:
        if not self.allow_exchange or (self.venue == 'mainnet' and not self.allow_mainnet):
            raise ValueError('Exchange execution is disabled by server policy')
        if not self.api_key or not self.api_secret:
            raise ValueError('Exchange execution requires venue credentials')


def load_launcher_environment() -> RuntimeEnvironment:
    # Explicit launchers only. Deployment environment wins over local .env.
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env', override=False)
    return RuntimeEnvironment.from_mapping(os.environ)


def construct_exchange(runtime: RuntimeEnvironment, *, private: bool = False) -> Any:
    import ccxt
    if private:
        runtime.require_exchange()
    exchange = ccxt.binance({
        'enableRateLimit': True, 'options': {'defaultType': 'spot'},
        **({'apiKey': runtime.api_key, 'secret': runtime.api_secret} if private else {}),
    })
    # Must be the first call: sandbox keys are distinct from production keys.
    if runtime.venue == 'testnet':
        exchange.set_sandbox_mode(True)
    return exchange
