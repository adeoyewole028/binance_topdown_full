"""Canonical manual request and stable financial review identity."""
from __future__ import annotations
from typing import Any
from datetime import datetime
import hashlib
import json
import re
from configuration import ConfigurationError
from paper_policy import number, decimal_text
from operations_policy import OperationsPolicy


def fresh_price(quote: dict[str,Any] | None, now: float) -> bool:
    if not quote:return False
    try:
        stamp=datetime.fromisoformat(quote['fetched_at'])
        return stamp.tzinfo is not None and 0<=now-stamp.timestamp()<OperationsPolicy().price_expiry
    except (ValueError,TypeError,KeyError):return False


def canonical_request(kind: str, symbol: str, side: str, quantity: str, sizing: str) -> dict[str,str]:
    if kind not in {'trade','close'} or side not in {'buy','sell'} or not isinstance(symbol,str) or not re.fullmatch(r'[A-Z0-9]{2,12}/[A-Z0-9]{2,12}',symbol) or (kind=='close' and side!='sell'):
        raise ConfigurationError('Invalid order side or canonical symbol')
    if sizing not in {'quantity','fixed','risk','full_close'} or (sizing in {'fixed','risk'} and side!='buy') or (sizing=='full_close' and (kind!='close' or side!='sell')):
        raise ConfigurationError('Invalid sizing mode')
    amount=number(quantity) if sizing=='quantity' else None
    if amount is not None and amount<=0:raise ConfigurationError('Order quantity must be positive')
    return {'kind':kind,'symbol':symbol,'side':side,'qty':decimal_text(amount) if amount is not None else '', 'sizing':sizing}


def financial_fingerprint(snapshot: dict[str,Any]) -> str:
    account=snapshot['paper_account'] if snapshot['config']['paper_trading'] else snapshot['exchange_account']
    values={'config_version':snapshot['config_version'],'prices':snapshot['market_prices'],
        'balances':account['balances'],'account_state':account['state'],
        'positions':snapshot['positions'],'operations':snapshot['operations'],
        'entry_control':snapshot['entry_control'],'equity':snapshot['performance']['equity'],
        'owner':{key:snapshot['ownership'].get(key) for key in ('state','generation','token')},
        'protection':[{key:group.get(key) for key in ('group_id','symbol','state','quantity','stop_price','take_profit_price')}
                      for group in snapshot['protection']]}
    return hashlib.sha256(json.dumps(values,sort_keys=True,allow_nan=False).encode()).hexdigest()
