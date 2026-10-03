"""Owned Spot OCO groups; ambiguity never authorizes a second sell."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
import json
import uuid
from typing import Any, TYPE_CHECKING

from exchange_adapter import SubmissionRejected
from execution_owner import OwnershipUnavailable
from paper_filters import validate_market_quantity
from paper_policy import decimal_text, financial_arithmetic, number
from request_budget import ProviderFailure
from state_store import StateUnavailable

if TYPE_CHECKING:
    from exchange_execution import ExchangeExecution


@financial_arithmetic
def protection_terms(metadata: dict[str, Any], symbol: str, quantity: Decimal, entry: Decimal,
                     config: dict[str, Any], now: float, venue: str) -> dict[str, str]:
    """Require known Spot capability and precision before admitting an entry."""
    market = next((m for m in metadata.get('symbols', []) if m['baseAsset']+'/'+m['quoteAsset'] == symbol), None)
    if not market or market.get('ocoAllowed') is not True or not {'TAKE_PROFIT', 'STOP_LOSS'}.issubset(market.get('orderTypes', [])):
        raise ValueError('spot_market_protection_unsupported')
    filters = {f['filterType']: f for f in market['filters']}
    price_rule = filters.get('PRICE_FILTER')
    if price_rule is None:
        raise ValueError('protection_price_rules_unavailable')
    step = number(filters['LOT_SIZE']['stepSize'])
    # A remainder below a quantity step is explicit unprotected dust, never an
    # over-sized protective sell. Whole-holding coverage is required here.
    if step <= 0 or quantity % step:
        raise ValueError('protection_quantity_review_required')
    tick = number(price_rule['tickSize'])
    minimum, maximum = number(price_rule['minPrice']), number(price_rule['maxPrice'])
    if tick <= 0 or min(minimum, maximum) < 0:
        raise ValueError('Invalid protection price filter')
    stop = (entry*(1-number(config['stop_loss_pct']))/tick).to_integral_value(rounding=ROUND_FLOOR)*tick
    target = (entry*(1+number(config['take_profit_pct']))/tick).to_integral_value(rounding=ROUND_FLOOR)*tick
    if not 0 < stop < entry < target or any(p < minimum or (maximum and p > maximum) for p in (stop, target)):
        raise ValueError('protection_price_filter_rejected')
    validate_market_quantity(metadata, symbol, quantity, entry, now, venue)
    return {'stop_price': decimal_text(stop), 'take_profit_price': decimal_text(target)}


class ExchangeProtection:
    def __init__(self, execution: ExchangeExecution) -> None:
        self.execution, self.owner, self.store = execution, execution.owner, execution.store

    def _owned(self, connection: Any) -> None:
        if not self.owner._matches_owner(connection):
            raise OwnershipUnavailable('Protection ownership uncertain')

    def _event(self, connection: Any, group: str, values: dict[str, Any]) -> None:
        self.execution._event(connection, 'protection_observed', values, group)

    def state(self, group: str, state: str, **values: Any) -> None:
        with self.store.connection(write=True) as connection:
            self._owned(connection)
            connection.execute('UPDATE protection_groups SET state=? WHERE namespace=? AND group_id=?', (state, self.store.namespace, group))
            self._event(connection, group, {'next_action': None if state in {'active', 'closed'} else
                         'Reconcile this protection list and its legs before any new entry or conflicting sell', **values})
            if state in {'failed', 'unknown'}:
                self.execution._event(connection, 'protection_alert', {'group_id': group, 'state': state, 'reason': values.get('reason', state)})

    def groups(self) -> list[dict[str, Any]]:
        with self.store.connection() as connection:
            return self.store._protection(connection)

    def group(self, identity: str) -> dict[str, Any]:
        group = next((g for g in self.groups() if g['group_id'] == identity), None)
        if group is None:
            raise StateUnavailable('Required protection group unavailable')
        return group

    def fail_unattempted(self, identity: str) -> None:
        if self.group(identity).get('dispatch_attempted') is not False:
            raise StateUnavailable('Protection dispatch outcome is uncertain')
        with self.store.connection(write=True) as connection:
            self._owned(connection)
            for row in connection.execute("SELECT operation_id FROM intents WHERE namespace=? AND request_hash=?", (self.store.namespace, 'protection:'+identity)):
                connection.execute("UPDATE intents SET state='rejected',error_class='protection_not_dispatched' WHERE namespace=? AND operation_id=?", (self.store.namespace, row[0]))
                connection.execute("UPDATE reservations SET state='released' WHERE namespace=? AND operation_id=?", (self.store.namespace, row[0]))
        self.state(identity, 'failed', reason='protection_request_not_attempted',
                   next_action='No protection call was sent. Review the unprotected holding and submit an authenticated reduction')

    def verify_all(self) -> None:
        for group in self.groups():
            if group['state'] in {'installing', 'unknown', 'active', 'canceling'}:
                if group['state'] in {'installing', 'unknown'} and group.get('dispatch_attempted') is False:
                    self.fail_unattempted(group['group_id'])
                else:
                    self.verify(group)

    def maintain(self) -> None:
        snapshot = self.store.snapshot(now=self.owner.clock())
        holdings = {p['symbol']: Decimal(p['qty']) for p in snapshot['positions']}
        for group in snapshot['protection']:
            held = holdings.get(group['symbol'], Decimal(0))
            if group['state'] == 'active' and held != Decimal(group['quantity']):
                # Only verified, owned protection is canceled. A new generation
                # must cover the complete confirmed holding after additional fills.
                self.prepare_close({'symbol': group['symbol'], 'requested_qty': '0', 'operation_id': group['group_id']})
            elif group['state'] == 'failed' and held == 0 and (group.get('dispatch_attempted') is False or group.get('rejected_before_engine')):
                self.state(group['group_id'], 'closed')
        self.install_missing()

    def verify(self, group: dict[str, Any]) -> bool:
        """Query both actual orders; a placement/cancel acknowledgement is insufficient."""
        observed = self.execution._request('reconciliation', 4, lambda: self.execution.provider.query_list(group['group_id']))
        if observed is None:
            self.state(group['group_id'], 'unknown', reason='provider_list_unresolved')
            return False
        expected = {group['above_id']: ('TAKE_PROFIT', group['take_profit_price']), group['below_id']: ('STOP_LOSS', group['stop_price'])}
        if (observed.get('symbol') != group['symbol'].replace('/', '') or observed.get('listClientOrderId') != group['group_id'] or
            observed.get('contingencyType') != 'OCO' or len(observed.get('orders', [])) != 2 or
            {r.get('clientOrderId') for r in observed.get('orders', [])} != set(expected)):
            raise StateUnavailable('Protection list identity mismatch')
        list_id = str(observed['orderListId'])
        if group.get('provider_list_id') is not None and group['provider_list_id'] != list_id:
            raise StateUnavailable('Protection list identity changed')
        states = []
        for client, (kind, stop) in expected.items():
            with self.store.connection() as connection:
                operation = dict(connection.execute('SELECT * FROM intents WHERE namespace=? AND operation_id=?', (self.store.namespace, client)).fetchone())
            leg = self.execution._request('reconciliation', 4, lambda: self.execution.query_operation(operation))
            if leg is None:
                self.state(group['group_id'], 'unknown', reason='provider_leg_unresolved')
                return False
            if leg.get('type') != kind or number(leg.get('stopPrice')) != number(stop) or str(leg.get('orderListId')) != list_id:
                raise StateUnavailable('Protection leg policy mismatch')
            self.execution.observe(operation, leg)
            with self.store.connection() as connection:
                states.append(connection.execute('SELECT state FROM intents WHERE namespace=? AND operation_id=?', (self.store.namespace, client)).fetchone()[0])
        terminal = all(s in {'filled', 'canceled', 'expired', 'rejected'} for s in states)
        active = states == ['accepted', 'accepted'] and observed.get('listOrderStatus') == 'EXECUTING' and group['state'] != 'canceling'
        state = 'closed' if terminal and observed.get('listOrderStatus') == 'ALL_DONE' else 'active' if active else 'canceling' if group['state'] == 'canceling' else 'unknown'
        with self.store.connection(write=True) as connection:
            self._owned(connection)
            connection.execute('UPDATE protection_groups SET provider_list_id=?,verified_at=?,state=? WHERE namespace=? AND group_id=?',
                               (list_id, datetime.fromtimestamp(self.owner.clock(), timezone.utc).isoformat(), state, self.store.namespace, group['group_id']))
            self._event(connection, group['group_id'], {'semantics': 'exchange_confirmed', 'leg_states': states,
                                                       'next_action': None if state in {'active', 'closed'} else 'Resolve both legs and list before a conflicting sell'})
        return state in {'active', 'closed'} or (state == 'unknown' and observed.get('listOrderStatus') == 'EXECUTING' and all(s in {'accepted', 'partially_filled', 'canceled', 'filled'} for s in states))

    @financial_arithmetic
    def install_missing(self) -> None:
        snapshot = self.store.snapshot(now=self.owner.clock())
        recoverable_partial = (snapshot['exchange_account'].get('identity') and
                               snapshot['exchange_account']['state'] == 'reconciliation_required' and
                               any(o['side'] == 'buy' and o['state'] in {'accepted', 'partially_filled'} for o in snapshot['operations']) and
                               not any(o['state'] in {'dispatching', 'unknown', 'fills_pending'} for o in snapshot['operations']))
        if snapshot['exchange_account']['state'] != 'ready' and not recoverable_partial:
            return
        for position in snapshot['positions']:
            symbol = position['symbol']
            if any(g['symbol'] == symbol and g['state'] != 'closed' for g in snapshot['protection']):
                continue
            # A serial manual close is serviced first; its confirmed remainder
            # receives the next generation after the close outcome is reconciled.
            if any(o['symbol'] == symbol and o['side'] == 'sell' and o['kind'] != 'protection' and o['state'] in {'queued', 'dispatching', 'unknown', 'accepted', 'partially_filled', 'fills_pending'} for o in snapshot['operations']):
                continue
            rule = snapshot['position_rules'].get(symbol, {})
            if rule.get('state') != 'bound':
                continue  # Unknown legacy basis/history cannot supply a policy.
            group, above, below = (uuid.uuid4().hex for _ in range(3))
            quantity, entry = Decimal(position['qty']), Decimal(position['entry'])
            if not quantity.is_finite() or not entry.is_finite() or min(quantity, entry) <= 0:
                raise StateUnavailable('Invalid confirmed protection holding')
            values: dict[str, Any] = {'semantics': 'exchange_pending', 'above_id': above, 'below_id': below,
                                      'config_version': rule['config_version'], 'installed_at': self.owner.clock(), 'dispatch_attempted': False}
            failure = None
            try:
                values.update(protection_terms(snapshot['scan'].get('market_metadata', {}), symbol, quantity, entry,
                                               rule['config'], self.owner.clock(), self.store.venue))
                quote = snapshot['market_prices'].get(symbol)
                if quote is None or not number(values['stop_price']) < number(quote['price']) < number(values['take_profit_price']):
                    raise ValueError('protection_trigger_already_crossed')
                available = number(snapshot['exchange_account']['balances'].get(symbol.split('/')[0], {}).get('available', '0'))
                if quantity > available:
                    raise ValueError('confirmed_sellable_quantity_unavailable')
            except ValueError as error:
                failure = str(error)
            with self.store.connection(write=True) as connection:
                self._owned(connection)
                entry_operation = connection.execute("SELECT operation_id FROM fills WHERE namespace=? AND symbol=? AND side='buy' ORDER BY rowid DESC LIMIT 1", (self.store.namespace, symbol)).fetchone()
                if entry_operation is None:
                    continue
                connection.execute('INSERT INTO protection_groups VALUES (?,?,?,?,?,?,?,?)',
                                   (self.store.namespace, group, symbol, entry_operation[0], decimal_text(quantity), 'failed' if failure else 'installing', None, None))
                self._event(connection, group, values)
                if failure is None:
                    for client in (above, below):
                        connection.execute('''INSERT INTO intents(namespace,operation_id,idempotency_key,request_hash,kind,symbol,side,requested_qty,config_version,state,provider_client_id)
                                              VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                                           (self.store.namespace, client, client, 'protection:'+group, 'protection', symbol, 'sell', decimal_text(quantity), rule['config_version'], 'prepared', client))
            if failure:
                self.state(group, 'failed', reason=failure)
                continue
            try:
                def prepare() -> None:
                    with self.store.connection(write=True) as connection:
                        self._owned(connection)
                        self._event(connection, group, {'dispatch_attempted': True})
                        for index, client in enumerate((above, below)):
                            amount = quantity if index == 0 else Decimal(0)
                            connection.execute("UPDATE intents SET state='dispatching' WHERE namespace=? AND operation_id=? AND state='prepared'", (self.store.namespace, client))
                            connection.execute('INSERT INTO reservations VALUES (?,?,?,?,?)', (self.store.namespace, client, symbol.split('/')[0], decimal_text(amount), 'active'))
                            self.execution._event(connection, 'exchange_dispatching', {'reservation_amount': decimal_text(amount), 'asset': symbol.split('/')[0], 'group_id': group}, client)
                self.execution._request('reduction', 1, lambda _: self.execution.provider.protect(symbol, decimal_text(quantity), values['stop_price'], values['take_profit_price'], group, above, below), prepare=prepare)
                current = self.group(group)
                self.verify(current)
            except SubmissionRejected as error:
                with self.store.connection(write=True) as connection:
                    self._owned(connection)
                    for client in (above, below):
                        connection.execute("UPDATE intents SET state='rejected',error_class='protection_rejected' WHERE namespace=? AND operation_id=?", (self.store.namespace, client))
                        connection.execute("UPDATE reservations SET state='released' WHERE namespace=? AND operation_id=?", (self.store.namespace, client))
                self.state(group, 'failed', reason=error.kind, rejected_before_engine=True)
            except (ProviderFailure, TimeoutError, ConnectionError):
                if self.group(group).get('dispatch_attempted') is False:
                    self.fail_unattempted(group)
                else:
                    self.state(group, 'unknown', reason='protection_dispatch_unknown')

    def prepare_close(self, operation: dict[str, Any]) -> bool:
        """Cancellation uncertainty retains reservations and blocks market sells."""
        for group in self.groups():
            if group['symbol'] != operation['symbol'] or group['state'] == 'closed':
                continue
            if group['state'] == 'failed' and group.get('rejected_before_engine'):
                continue
            if group['state'] == 'failed' and group.get('dispatch_attempted') is False:
                continue  # Local validation failed without any provider dispatch.
            was_canceling = group['state'] == 'canceling'
            if not self.verify(group):
                return False
            current = self.group(group['group_id'])
            if current['state'] == 'closed':
                continue
            if was_canceling:
                return False  # Never repeat an uncertain cancel as permission to sell.
            cancel_id = uuid.uuid4().hex
            try:
                self.execution._request('reduction', 1, lambda _: self.execution.provider.cancel_list(group['symbol'], group['group_id'], cancel_id),
                                        prepare=lambda: self.state(group['group_id'], 'canceling', cancel_id=cancel_id))
            except (ProviderFailure, TimeoutError, ConnectionError):
                return False
            current = self.group(group['group_id'])
            self.verify(current)
            if self.group(group['group_id'])['state'] != 'closed':
                return False
        self.execution.reconcile()
        snapshot = self.store.snapshot(now=self.owner.clock())
        holding = next((p for p in snapshot['positions'] if p['symbol'] == operation['symbol']), None)
        remaining = number(holding['qty']) if holding else Decimal(0)
        with self.store.connection() as connection:
            queued = connection.execute("SELECT payload FROM journal WHERE namespace=? AND operation_id=? AND event='intent_queued' LIMIT 1", (self.store.namespace, operation['operation_id'])).fetchone()
        full_close = queued is not None and json.loads(queued[0]).get('sizing') == 'full_close'
        if remaining == 0 or (number(operation['requested_qty']) > remaining and not full_close):
            # Preserve the reviewed immutable amount; a changed holding requires
            # a fresh operator review instead of silently changing its meaning.
            with self.store.connection(write=True) as connection:
                self._owned(connection)
                connection.execute("UPDATE intents SET state='rejected',error_class='holding_changed_review_required' WHERE namespace=? AND operation_id=?", (self.store.namespace, operation['operation_id']))
            return False
        if full_close:
            with self.store.connection(write=True) as connection:
                self._owned(connection)
                self.execution._event(connection, 'close_reconciled_quantity', {'quantity': decimal_text(min(remaining, number(operation['requested_qty'])))}, operation['operation_id'])
        return bool(snapshot['exchange_account']['state'] == 'ready')
