"""Pinned Spot market-order rules; unknown provider inputs fail closed."""
from __future__ import annotations
from datetime import datetime
from decimal import Decimal
from typing import Any
from operations_policy import OperationsPolicy
from paper_policy import number


def validate_position_limit(metadata: dict[str, Any], symbol: str, projected_base: Decimal) -> None:
    """MAX_POSITION counts all confirmed base assets and outstanding buys."""
    for market in metadata.get('symbols', []):
        if market['baseAsset'] + '/' + market['quoteAsset'] != symbol:
            continue
        for rule in market['filters']:
            if rule['filterType'] == 'MAX_POSITION':
                maximum = number(rule['maxPosition'])
                if maximum < 0 or projected_base > maximum:
                    raise ValueError('position_filter_rejected')


def validate_market_quantity(metadata: dict[str, Any], symbol: str, quantity: Decimal,
                             last_price: Decimal, now: float, venue: str) -> None:
    try:
        stamp = datetime.fromisoformat(metadata['fetched_at'])
        if metadata['venue'] != venue or stamp.tzinfo is None or not 0 <= now - stamp.timestamp() < OperationsPolicy().scan_health_timeout:
            raise ValueError('market_metadata_stale')
        market = next(s for s in metadata['symbols'] if s['baseAsset'] + '/' + s['quoteAsset'] == symbol)
        if market['status'] != 'TRADING' or market['isSpotTradingAllowed'] is not True or 'MARKET' not in market['orderTypes']:
            raise ValueError('market_not_tradable')
        filters = {f['filterType']: f for f in market['filters']}
        if 'LOT_SIZE' not in filters:
            raise ValueError('quantity_rules_unavailable')
        for kind in ('LOT_SIZE', 'MARKET_LOT_SIZE'):
            if kind not in filters:
                continue
            rule = filters[kind]
            minimum, maximum, step = (number(rule[k]) for k in ('minQty', 'maxQty', 'stepSize'))
            if min(minimum, maximum, step) < 0 or (maximum and minimum > maximum):
                raise ValueError('invalid_quantity_rules')
            if quantity < minimum or (maximum and quantity > maximum) or (step and quantity % step):
                raise ValueError('quantity_filter_rejected')
        for kind in ('MIN_NOTIONAL', 'NOTIONAL'):
            if kind not in filters:
                continue
            rule = filters[kind]
            lower = rule.get('applyToMarket', rule.get('applyMinToMarket', False))
            upper = rule.get('applyMaxToMarket', False)
            if type(lower) is not bool or type(upper) is not bool:
                raise ValueError('invalid_notional_flags')
            if not lower and not upper:
                continue
            observed = metadata['filter_prices'][symbol]
            reference_stamp = datetime.fromisoformat(observed['fetched_at'])
            if reference_stamp.tzinfo is None or not 0 <= now - reference_stamp.timestamp() < OperationsPolicy().price_expiry:
                raise ValueError('notional_reference_stale')
            reference = observed['reference'].get('referencePrice')
            if reference is not None:
                filter_price = number(reference)
            elif rule['avgPriceMins'] == 0:
                filter_price = last_price
            elif observed['average']['mins'] == rule['avgPriceMins']:
                filter_price = number(observed['average']['price'])
            else:
                raise ValueError('notional_reference_unavailable')
            if filter_price <= 0:
                raise ValueError('invalid_filter_price')
            if (lower and quantity * filter_price < number(rule['minNotional'])) or (upper and quantity * filter_price > number(rule['maxNotional'])):
                raise ValueError('notional_filter_rejected')
    except (KeyError, StopIteration, TypeError) as error:
        raise ValueError('market_rules_unavailable') from error
