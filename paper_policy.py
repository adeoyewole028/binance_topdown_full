"""Explicit versioned simulation assumptions; no numerical financial defaults."""
from __future__ import annotations
from decimal import Context, Decimal, InvalidOperation, localcontext
from functools import wraps
import re
from typing import Any, Callable, ParamSpec, TypeVar
from configuration import ConfigurationError

DECIMAL_FIELDS = {'initial_cash', 'fee_fraction', 'slippage_fraction', 'risk_fraction', 'fixed_quote',
                  'max_allocation_fraction', 'max_exposure_fraction', 'drawdown_fraction'}
INTEGER_FIELDS = {'max_positions', 'cooldown_seconds'}
FIELDS = DECIMAL_FIELDS | INTEGER_FIELDS | {'quote'}
P = ParamSpec('P')
T = TypeVar('T')


def financial_arithmetic(function: Callable[P, T]) -> Callable[P, T]:
    """Keep accepted 28-digit inputs and compounded fees exact in money effects.

    Division for sizing/basis has deterministic 256-digit precision; monetary
    products, sums and serialization must not inherit a caller's short context.
    """
    @wraps(function)
    def calculate(*args: P.args, **kwargs: P.kwargs) -> T:
        with localcontext(Context(prec=256)):
            return function(*args, **kwargs)
    return calculate


def number(value: Any) -> Decimal:
    try:
        if isinstance(value, bool):
            raise ValueError('Boolean money value')
        result = Decimal(str(value))
        significant_digits = len(''.join(str(digit) for digit in result.as_tuple().digits).rstrip('0'))
        if not result.is_finite() or (result and (abs(result.adjusted()) > 28 or significant_digits > 28)):
            raise ValueError('Nonfinite money value')
        return result
    except (InvalidOperation, ValueError, TypeError) as error:
        raise ConfigurationError('Finite decimal value required') from error


def validated_policy(values: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(values, dict) or not FIELDS.issubset(values):
        raise ConfigurationError('Complete explicit paper setup required')
    result: dict[str, Any] = {key: decimal_text(number(values[key])) for key in DECIMAL_FIELDS}
    for key in INTEGER_FIELDS:
        value = number(values[key])
        if value != value.to_integral_value() or value < (1 if key == 'max_positions' else 0):
            raise ConfigurationError('Invalid paper capacity or cooldown')
        result[key] = int(value)
    quote = values['quote']
    if not isinstance(quote, str) or not re.fullmatch(r'[A-Z0-9]{2,12}', quote):
        raise ConfigurationError('Canonical quote asset required')
    result['quote'] = quote
    for key in DECIMAL_FIELDS:
        value = number(result[key])
        if key in {'fee_fraction', 'slippage_fraction'}:
            valid = 0 <= value < 1
        elif key in {'initial_cash', 'fixed_quote'}:
            valid = value > 0
        else:
            valid = 0 < value <= 1
        if not valid:
            raise ConfigurationError('Paper assumptions outside supported range')
    result['model'] = 'paper-full-fill-weighted-average-v1'
    return result


def decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip('0').rstrip('.') if '.' in text else text
