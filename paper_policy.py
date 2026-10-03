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
    return _validated_policy(values, paper=True)


def validated_execution_policy(values: dict[str, Any]) -> dict[str, Any]:
    return _validated_policy(values, paper=False)


def _validated_policy(values: dict[str, Any], *, paper: bool) -> dict[str, Any]:
    required = FIELDS if paper else FIELDS - {'initial_cash'}
    decimals = DECIMAL_FIELDS if paper else DECIMAL_FIELDS - {'initial_cash'}
    if not isinstance(values, dict) or not required.issubset(values):
        raise ConfigurationError('Complete explicit operator setup required')
    result: dict[str, Any] = {key: decimal_text(number(values[key])) for key in decimals}
    for key in INTEGER_FIELDS:
        value = number(values[key])
        if value != value.to_integral_value() or value < (1 if key == 'max_positions' else 0):
            raise ConfigurationError('Invalid capacity or cooldown')
        result[key] = int(value)
    quote = values['quote']
    if not isinstance(quote, str) or not re.fullmatch(r'[A-Z0-9]{2,12}', quote):
        raise ConfigurationError('Canonical quote asset required')
    result['quote'] = quote
    for key in decimals:
        value = number(result[key])
        if key in {'fee_fraction', 'slippage_fraction'}:
            valid = 0 <= value < 1
        elif key in {'initial_cash', 'fixed_quote'}:
            valid = value > 0
        else:
            valid = 0 < value <= 1
        if not valid:
            raise ConfigurationError('Operator assumptions outside supported range')
    result['model'] = 'paper-full-fill-weighted-average-v1' if paper else 'spot-observed-weighted-average-v1'
    return result


def decimal_text(value: Decimal) -> str:
    if value == 0:
        return '0'
    text = format(value, "f")
    return text.rstrip('0').rstrip('.') if '.' in text else text


@financial_arithmetic
def fill_terms(quantity: Decimal, market_price: Decimal, side: str, policy: dict[str,Any]) -> dict[str,Decimal]:
    """Shared full-fill price/cost/quote fee; filtering precedes accounting."""
    if side not in {'buy','sell'} or min(quantity,market_price)<=0:
        raise ConfigurationError('Positive simulated quantity/price and valid side required')
    price=market_price*(1+Decimal(policy['slippage_fraction'])*(1 if side=='buy' else -1))
    cost=quantity*price
    return {'price':price,'cost':cost,'fee':cost*Decimal(policy['fee_fraction'])}


@financial_arithmetic
def entry_quantity(*, cash: Decimal, equity: Decimal, exposure: Decimal, market_price: Decimal,
                   policy: dict[str,Any], config: dict[str,Any], mode: str) -> Decimal:
    """Same currency-qualified capital/allocation/stop-risk sizing in all paths."""
    if mode not in {'fixed','risk'} or market_price<=0 or min(cash,equity,exposure)<0:
        raise ConfigurationError('Valid capital, price and fixed/risk sizing required')
    price=market_price*(1+Decimal(policy['slippage_fraction']))
    unit_cost=price*(1+Decimal(policy['fee_fraction']))
    budget=min(cash,equity*Decimal(policy['max_allocation_fraction']),equity*Decimal(policy['max_exposure_fraction'])-exposure)
    if mode=='fixed':budget=min(budget,Decimal(policy['fixed_quote']))
    quantity=budget/unit_cost
    if mode=='risk':
        loss=price*(Decimal(str(config['stop_loss_pct']))+2*Decimal(policy['fee_fraction'])+Decimal(policy['slippage_fraction']))
        quantity=min(quantity,equity*min(Decimal(policy['risk_fraction']),Decimal(str(config['risk'])))/loss)
    return quantity
