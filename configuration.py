"""Complete configuration validation; financial risk limits remain D05 work."""
from __future__ import annotations

import math
import re
from typing import Any

from runtime_environment import parse_boolean


class ConfigurationError(RuntimeError):
    """Invalid values cannot produce a partly saved version."""


class ConfigurationConflict(ConfigurationError):
    """Review the newer configuration before overwriting it."""


NUMERIC_FIELDS = {'risk', 'near_ema_pct', 'rsi_threshold', 'rsi_exec_threshold', 'rsi_overbought',
                  'take_profit_pct', 'stop_loss_pct', 'trailing_pct', 'min_quote_vol', 'equity_proxy'}
INTEGER_FIELDS = {'max_positions', 'cooldown', 'universe_size'}
BOOLEAN_FIELDS = {'paper_trading', 'use_dynamic_universe'}
FIELDS = NUMERIC_FIELDS | INTEGER_FIELDS | BOOLEAN_FIELDS | {'base_quote'}


def validated_config(values: dict[str, Any]) -> dict[str, Any]:
    if not FIELDS.issubset(values):
        raise ConfigurationError('Complete configuration required')
    result = values.copy()
    try:
        for key in NUMERIC_FIELDS | INTEGER_FIELDS:
            value = result[key]
            if isinstance(value, bool):
                raise ConfigurationError(f'Invalid {key}')
            number = float(value)
            if not math.isfinite(number):
                raise ConfigurationError(f'Invalid {key}')
            if key in INTEGER_FIELDS:
                if not number.is_integer():
                    raise ConfigurationError(f'{key} must be an integer')
                result[key] = int(number)
            else:
                result[key] = number
        for key in BOOLEAN_FIELDS:
            value = result[key]
            if not isinstance(value, (bool, str)):
                raise ConfigurationError(f'Invalid {key}')
            result[key] = value if isinstance(value, bool) else parse_boolean(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ConfigurationError('Invalid configuration value') from error
    positive = {'risk', 'max_positions', 'universe_size', 'take_profit_pct', 'stop_loss_pct', 'equity_proxy'}
    nonnegative = {'cooldown', 'min_quote_vol', 'near_ema_pct', 'trailing_pct'}
    if any(result[key] <= 0 for key in positive) or any(result[key] < 0 for key in nonnegative):
        raise ConfigurationError('Configuration outside supported range')
    if result['risk'] > 1 or result['stop_loss_pct'] >= 1 or result['near_ema_pct'] >= 1 or result['trailing_pct'] >= 1:
        raise ConfigurationError('Configuration outside supported range')
    if any(not 0 <= result[key] <= 100 for key in ('rsi_threshold', 'rsi_exec_threshold', 'rsi_overbought')):
        raise ConfigurationError('RSI thresholds must be between 0 and 100')
    if result['rsi_exec_threshold'] >= result['rsi_overbought']:
        raise ConfigurationError('Execution RSI must be below overbought RSI')
    quote = result['base_quote']
    if not isinstance(quote, str) or not re.fullmatch(r'[A-Z0-9]{2,12}', quote):
        raise ConfigurationError('Invalid quote asset')
    if 'paper_policy' in result:
        from paper_policy import validated_policy
        result['paper_policy'] = validated_policy(result['paper_policy'])
    if 'execution_policy' in result:
        from paper_policy import validated_execution_policy
        result['execution_policy'] = validated_execution_policy(result['execution_policy'])
    return result
