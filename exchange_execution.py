"""Owned Spot dispatch; network outcomes and financial facts are separate."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import re
from typing import Any, TYPE_CHECKING

from configuration import ConfigurationError
from paper_policy import decimal_text, financial_arithmetic, number
from paper_filters import validate_market_quantity, validate_position_limit
from state_store import StateUnavailable
from execution_owner import OwnershipUnavailable
from request_budget import RequestBudget, ProviderFailure
from exchange_adapter import SubmissionRejected

if TYPE_CHECKING:
    from execution_worker import ExecutionWorker


class ExchangeExecution:
    def __init__(self, owner: ExecutionWorker, provider: Any) -> None:
        self.owner, self.store, self.provider = owner, owner.store, provider
        self.budget = RequestBudget(self.store, clock=owner.clock, reserve_weight=172)
        self.last_reconcile_attempt: float | None = None
        from exchange_protection import ExchangeProtection
        self.protection = ExchangeProtection(self)

    def _event(self, connection: Any, event: str, values: dict[str, Any], operation: str | None = None) -> None:
        connection.execute('INSERT INTO journal(namespace,operation_id,event,payload) VALUES (?,?,?,?)',
                           (self.store.namespace, operation, event, json.dumps(values, sort_keys=True, allow_nan=False)))

    def _health(self, state: str, reason: str | None = None) -> None:
        with self.store.connection(write=True) as connection:
            if not self.owner._matches_owner(connection):
                raise OwnershipUnavailable('Exchange owner uncertain')
            connection.execute('UPDATE execution_owners SET state=? WHERE token=? AND generation=?',
                               ('ready' if state == 'ready' else 'recovering', self.owner.token, self.owner.generation))
            self._event(connection, 'exchange_health', {'state': state, 'reason': reason,
                                                       'as_of': self.owner.clock(), 'owner_generation': self.owner.generation})
            if state == 'review_required':
                self._event(connection, 'exchange_review_required', {'state': state, 'reason': reason})

    def _request(self, lane: str, weight: int, callback: Any, *, prepare: Any = None) -> Any:
        if not self.owner.ownership_confirmed():
            raise OwnershipUnavailable('Exchange owner uncertain')
        attempted = False
        def transport() -> Any:
            nonlocal attempted
            prepared = prepare() if prepare is not None else None
            attempted = True
            return callback(prepared) if prepare is not None else callback()
        try:
            return self.budget.request(lane, weight, transport)
        finally:
            if attempted:
                self.budget.observe(getattr(self.provider, 'headers', {}))

    def query_operation(self, operation: dict[str, Any]) -> dict[str, Any] | None:
        if operation['kind'] == 'protection' and operation.get('provider_order_id'):
            observed = self.provider.query_order(operation['symbol'], operation['provider_order_id'])
            if observed is not None and observed.get('status') == 'CANCELED' and str(observed.get('orderId')) == operation['provider_order_id']:
                # Binance may replace a canceled order's client ID. The already
                # verified immutable provider ID identifies this owned leg.
                observed = {**observed, 'clientOrderId': operation['provider_client_id']}
            return dict(observed) if observed is not None else None
        value = self.provider.query(operation['symbol'], operation['provider_client_id'])
        return dict(value) if value is not None else None

    def dispatch_quantity(self, operation: dict[str, Any]) -> Decimal:
        with self.store.connection() as connection:
            row = connection.execute("SELECT payload FROM journal WHERE namespace=? AND operation_id=? AND event='close_reconciled_quantity' ORDER BY sequence DESC LIMIT 1", (self.store.namespace, operation['operation_id'])).fetchone()
        quantity = number(json.loads(row[0])['quantity']) if row else number(operation['requested_qty'])
        if not 0 < quantity <= number(operation['requested_qty']):
            raise StateUnavailable('Invalid reconciled close quantity')
        return quantity

    @financial_arithmetic
    def reconcile(self) -> None:
        if not self.owner.ownership_confirmed():
            raise OwnershipUnavailable('Exchange owner uncertain')
        self.last_reconcile_attempt = self.owner.clock()
        self._health('reconciliation_required')
        with self.store.connection() as connection:
            outside = connection.execute("""SELECT 1 FROM intents i JOIN namespaces n ON n.id=i.namespace
                WHERE n.account=? AND n.venue=? AND i.namespace!=?
                AND i.state IN ('dispatching','unknown','accepted','partially_filled','fills_pending','canceling','processing') LIMIT 1""",
                (self.store.account, self.store.venue, self.store.namespace)).fetchone()
        if outside is not None:
            self._health('review_required', 'other_namespace_unresolved')
            return
        if self.provider.venue != self.store.venue:
            self._health('review_required', 'venue_identity_mismatch')
            return
        snapshot = self.store.snapshot()
        self.provider.configure(snapshot['scan'].get('market_metadata', {}))
        initial_account = self._request('reconciliation', 20, self.provider.account)
        identity = self._account_identity(initial_account)
        if identity is None:
            return
        if snapshot['exchange_account'].get('identity') and snapshot['exchange_account']['identity'] != identity:
            self._health('review_required', 'account_identity_mismatch')
            return
        self.protection.verify_all()
        with self.store.connection() as connection:
            identities = {row['provider_client_id']: dict(row) for row in connection.execute('SELECT * FROM intents WHERE namespace=?', (self.store.namespace,))}
        open_orders = self._request('reconciliation', 80, self.provider.open_orders)
        for observed in open_orders:
            operation = identities.get(observed.get('clientOrderId'))
            if operation is None or operation['state'] in {'queued', 'waiting_for_setup', 'rejected'}:
                self._health('review_required', 'unexpected_open_order')
                return
            self.observe(operation, observed)
        # A free kernel lock never authorizes a repeated private submission.
        for operation in snapshot['operations']:
            if operation['state'] in {'dispatching', 'unknown', 'accepted', 'partially_filled', 'fills_pending'}:
                with self.store.connection() as connection:
                    row = dict(connection.execute('SELECT * FROM intents WHERE namespace=? AND operation_id=?',
                                                  (self.store.namespace, operation['operation_id'])).fetchone())
                observed = self._request('reconciliation', 4, lambda: self.provider.query(row['symbol'], row['provider_client_id']))
                if observed is None:
                    self._health('reconciliation_required', 'provider_order_unresolved')
                    return
                self.observe(row, observed)
        history = self._history(snapshot, identities)
        if history is None:
            return
        account = self._request('reconciliation', 20, self.provider.account)
        final_identity = self._account_identity(account)
        if final_identity is None:
            return
        if final_identity != identity:
            self._health('review_required', 'account_identity_mismatch')
            return
        balances = {}
        for row in account['balances']:
            asset = row.get('asset')
            if not isinstance(asset, str) or re.fullmatch(r'[A-Z0-9]{1,20}', asset) is None or asset in balances:
                raise StateUnavailable('Invalid provider balance asset')
            free, locked = number(row['free']), number(row['locked'])
            if min(free, locked) < 0:
                raise StateUnavailable('Invalid provider balances')
            balances[asset] = {'free': decimal_text(free), 'locked': decimal_text(locked)}
        with self.store.connection(write=True) as connection:
            if not self.owner._matches_owner(connection):
                raise OwnershipUnavailable('Exchange owner uncertain')
            previous = self.store._exchange_account(connection)
            if previous.get('identity') and previous['identity'] != identity:
                self._event(connection, 'exchange_review_required', {'state': 'review_required', 'reason': 'account_identity_mismatch'})
                return
            if previous.get('identity'):
                changed = [asset for asset in set(previous['balances']) | set(balances)
                           if Decimal(previous['balances'].get(asset, {}).get('total', '0')) !=
                           sum((Decimal(v) for v in balances.get(asset, {'free': '0', 'locked': '0'}).values()), Decimal(0))]
                if changed:
                    self._event(connection, 'exchange_review_required', {'state': 'review_required', 'reason': 'unexpected_balance_change', 'assets': sorted(changed)})
                    return
            sequence = connection.execute('SELECT coalesce(max(rowid),0) FROM balance_movements WHERE namespace=?', (self.store.namespace,)).fetchone()[0]
            if not previous.get('identity'):
                config = self.store._config(connection)[1]
                quote = config.get('execution_policy', {}).get('quote', config['base_quote'])
                for asset, balance in balances.items():
                    quantity = Decimal(balance['free'])+Decimal(balance['locked'])
                    if asset != quote and quantity > 0:
                        connection.execute('INSERT INTO positions VALUES (?,?,?,?,?)',
                                           (self.store.namespace, asset+'/'+quote, decimal_text(quantity), '0', 'provider_basis_unknown'))
            self._event(connection, 'exchange_account_observed', {'identity': identity, 'venue': self.store.venue,
                                                                'balances': balances, 'movement_sequence': sequence,
                                                                'observed_at': self.owner.clock()})
            for symbol, values in history.items():
                self._event(connection, 'exchange_history_cursor', {'symbol': symbol, 'next_id': values['next_id']})
                if values['baseline_ids'] is not None:
                    self._event(connection, 'exchange_history_baseline', {'symbol': symbol, 'ids': values['baseline_ids']})
            self.store.record_exchange_risk(connection, now=self.owner.clock())
            unresolved = connection.execute("SELECT 1 FROM intents WHERE namespace=? AND (state IN ('dispatching','unknown','fills_pending') OR (kind!='protection' AND state IN ('accepted','partially_filled'))) LIMIT 1", (self.store.namespace,)).fetchone()
            review = connection.execute("SELECT 1 FROM journal WHERE namespace=? AND event='exchange_review_required' LIMIT 1", (self.store.namespace,)).fetchone()
            self._event(connection, 'exchange_health', {'state': 'review_required' if review else 'reconciliation_required' if unresolved else 'ready',
                                                       'reason': 'provider_order_unresolved' if unresolved else None,
                                                       'as_of': self.owner.clock(), 'owner_generation': self.owner.generation})
            connection.execute('UPDATE execution_owners SET state=? WHERE token=? AND generation=?',
                               ('recovering' if unresolved or review else 'ready', self.owner.token, self.owner.generation))

    def _account_identity(self, account: dict[str, Any]) -> str | None:
        if not account.get('uid') or account.get('canTrade') is not True or account.get('accountType') != 'SPOT' or 'SPOT' not in account.get('permissions', []):
            self._health('review_required', 'account_identity_or_permissions')
            return None
        return hashlib.sha256(str(account['uid']).encode()).hexdigest()

    def reconcile_if_due(self) -> None:
        if self.last_reconcile_attempt is not None and 0 <= self.owner.clock()-self.last_reconcile_attempt < self.owner.policy.reconciliation_interval:
            return
        try:
            self.reconcile()
        except ProviderFailure as error:
            self._health('reconciliation_required', error.kind)

    def _safe_reconcile(self) -> bool:
        try:
            self.reconcile()
            return True
        except (ProviderFailure, TimeoutError, ConnectionError) as error:
            self._health('reconciliation_required', getattr(error, 'kind', 'transient'))
            return False

    def _trades(self, symbol: str, *, cursor: int = 0, order_id: str | None = None) -> list[dict[str, Any]]:
        trades: list[dict[str, Any]] = []
        while True:
            page = self._request('reconciliation', 5 if order_id is not None else 20,
                                 lambda: self.provider.trades(symbol, from_id=cursor, order_id=order_id))
            ids = [fill.get('id') for fill in page]
            if len(page) > 1000 or any(type(identity) is not int or identity < cursor for identity in ids) or len(set(ids)) != len(ids):
                raise StateUnavailable('Invalid provider trade page')
            trades.extend(page)
            if len(page) < 1000:
                return trades
            cursor = max(ids) + 1

    def _history(self, snapshot: dict[str, Any], identities: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]] | None:
        with self.store.connection() as connection:
            cursors = {}
            baselines = {}
            identities = {row['provider_client_id']: dict(row) for row in connection.execute('SELECT * FROM intents WHERE namespace=?', (self.store.namespace,))}
            for row in connection.execute("SELECT payload FROM journal WHERE namespace=? AND event='exchange_history_cursor' ORDER BY sequence", (self.store.namespace,)):
                values = json.loads(row[0])
                if type(values['next_id']) is not int or values['next_id'] < 0:
                    raise StateUnavailable('Invalid durable trade cursor')
                cursors[values['symbol']] = values['next_id']
            for row in connection.execute("SELECT payload FROM journal WHERE namespace=? AND event='exchange_history_baseline' ORDER BY sequence", (self.store.namespace,)):
                values = json.loads(row[0])
                if not isinstance(values['ids'], list) or any(type(identity) is not int or identity < 0 for identity in values['ids']):
                    raise StateUnavailable('Invalid durable history baseline')
                baselines[values['symbol']] = set(values['ids'])
        symbols = set(snapshot['market_prices']) | {p['symbol'] for p in snapshot['positions']} | {o['symbol'] for o in identities.values()} | set(cursors)
        by_order = {(o['symbol'], str(o['provider_order_id'])): o for o in identities.values() if o['provider_order_id']}
        result = {}
        for symbol in sorted(symbols):
            next_id = cursors.get(symbol, 0)
            if symbol in cursors and symbol not in baselines:
                self._health('review_required', 'history_baseline_review_required')
                return None
            # A numeric trade ID is not an account-trade count. Complete,
            # budgeted overlap retains arbitrarily sparse/delayed identities.
            trades = self._trades(symbol, cursor=0)
            observed_orders: set[str] = set()
            for trade in trades:
                if trade['symbol'] != symbol.replace('/', ''):
                    raise StateUnavailable('Provider trade symbol mismatch')
                operation = by_order.get((symbol, str(trade['orderId'])))
                if operation is None:
                    if snapshot['exchange_account'].get('identity') and trade['id'] not in baselines.get(symbol, set()):
                        self._health('review_required', 'unexpected_trade_history')
                        return None
                    continue  # Pre-baseline history has no invented cost basis.
                if operation['operation_id'] not in observed_orders:
                    order = self._request('reconciliation', 4, lambda: self.query_operation(operation))
                    if order is None:
                        self._health('reconciliation_required', 'provider_order_unresolved')
                        return None
                    self.observe(operation, order)
                    observed_orders.add(operation['operation_id'])
            result[symbol] = {'next_id': max(next_id, max((trade['id']+1 for trade in trades), default=0)),
                              'baseline_ids': [trade['id'] for trade in trades] if symbol not in baselines else None}
        return result

    @financial_arithmetic
    def observe(self, operation: dict[str, Any], observed: dict[str, Any]) -> None:
        symbol, side = operation['symbol'], operation['side']
        if observed['symbol'] != symbol.replace('/', '') or observed['clientOrderId'] != operation['provider_client_id'] or observed['side'] != side.upper():
            raise StateUnavailable('Provider order identity mismatch')
        if operation.get('provider_order_id') and str(operation['provider_order_id']) != str(observed['orderId']):
            raise StateUnavailable('Provider order identity changed')
        requested = self.dispatch_quantity(operation)
        if number(observed['origQty']) != requested:
            raise StateUnavailable('Provider order quantity mismatch')
        trades = self._trades(symbol, order_id=str(observed['orderId']))
        namespace = self.store.namespace
        with self.store.connection(write=True) as connection:
            if not self.owner._matches_owner(connection):
                raise OwnershipUnavailable('Exchange owner uncertain')
            payload = {k: observed[k] for k in ('orderId', 'clientOrderId', 'symbol', 'side', 'status', 'executedQty', 'cummulativeQuoteQty', 'updateTime')}
            observation_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            connection.execute('INSERT OR IGNORE INTO provider_observations VALUES (?,?,?,?,?,?)',
                               (namespace, observation_id, operation['operation_id'], payload['status'], json.dumps(payload),
                                datetime.fromtimestamp(self.owner.clock(), timezone.utc).isoformat()))
            for fill in trades:
                if str(fill['orderId']) != str(observed['orderId']) or fill['symbol'] != symbol.replace('/', '') or fill['isBuyer'] is not (side == 'buy'):
                    raise StateUnavailable('Provider fill identity mismatch')
                identity = symbol + ':' + str(fill['id'])
                quantity, price, cost, fee = (number(fill[k]) for k in ('qty', 'price', 'quoteQty', 'commission'))
                if min(quantity, price, cost) <= 0 or fee < 0:
                    raise StateUnavailable('Invalid provider fill')
                base, quote = symbol.split('/')
                fee_asset = fill['commissionAsset']
                if not isinstance(fee_asset, str) or re.fullmatch(r'[A-Z0-9]{1,20}', fee_asset) is None:
                    raise StateUnavailable('Invalid provider fee asset')
                existing = connection.execute('SELECT * FROM fills WHERE namespace=? AND fill_id=?', (namespace, identity)).fetchone()
                if existing is not None:
                    if (existing['operation_id'] != operation['operation_id'] or existing['symbol'] != symbol or existing['side'] != side or
                        existing['fee_asset'] != fee_asset or any(Decimal(existing[key]) != value for key, value in
                                                               (('qty', quantity), ('price', price), ('quote_cost', cost), ('fee_amount', fee)))):
                        raise StateUnavailable('Provider fill identity changed')
                    continue
                position = connection.execute('SELECT qty,entry FROM positions WHERE namespace=? AND symbol=?', (namespace, symbol)).fetchone()
                held, entry = (Decimal(position['qty']), Decimal(position['entry'])) if position else (Decimal(0), Decimal(0))
                delta = quantity * (1 if side == 'buy' else -1) - (fee if fee_asset == base else Decimal(0))
                remaining = held + delta
                if remaining < 0:
                    raise StateUnavailable('Provider fill exceeds confirmed holdings')
                if side == 'buy' and remaining:
                    entry = (held*entry + cost + (fee if fee_asset == quote else Decimal(0))) / remaining
                if remaining:
                    connection.execute('INSERT INTO positions VALUES (?,?,?,?,?) ON CONFLICT(namespace,symbol) DO UPDATE SET qty=excluded.qty,entry=excluded.entry',
                                       (namespace, symbol, decimal_text(remaining), decimal_text(entry), 'spot-observed-weighted-average-v1'))
                else:
                    connection.execute('DELETE FROM positions WHERE namespace=? AND symbol=?', (namespace, symbol))
                if fee > 0 and fee_asset not in {base, quote}:
                    fee_symbol = fee_asset+'/'+quote
                    fee_position = connection.execute('SELECT qty FROM positions WHERE namespace=? AND symbol=?', (namespace, fee_symbol)).fetchone()
                    if fee_position is None or Decimal(fee_position['qty']) < fee:
                        raise StateUnavailable('Provider fee exceeds confirmed fee-asset holding')
                    fee_remaining = Decimal(fee_position['qty'])-fee
                    if fee_remaining:
                        connection.execute('UPDATE positions SET qty=? WHERE namespace=? AND symbol=?',
                                           (decimal_text(fee_remaining), namespace, fee_symbol))
                    else:
                        connection.execute('DELETE FROM positions WHERE namespace=? AND symbol=?', (namespace, fee_symbol))
                connection.execute('INSERT INTO fills(namespace,fill_id,operation_id,symbol,side,qty,price,quote_cost,fee_amount,fee_asset,profile) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                                   (namespace, identity, operation['operation_id'], symbol, side, decimal_text(quantity), decimal_text(price), decimal_text(cost), decimal_text(fee), fee_asset, 'spot-observed-weighted-average-v1'))
                movements = {base: quantity if side == 'buy' else -quantity, quote: -cost if side == 'buy' else cost}
                movements[fee_asset] = movements.get(fee_asset, Decimal(0)) - fee
                for asset, amount in movements.items():
                    connection.execute('INSERT INTO balance_movements(namespace,movement_id,asset,amount,event) VALUES (?,?,?,?,?)',
                                       (namespace, identity + ':' + asset, asset, decimal_text(amount), 'exchange_fill'))
                self._event(connection, 'exchange_fill', {'fill_id': identity, 'fee_asset': fee_asset}, operation['operation_id'])
                if side == 'buy':
                    stamp = datetime.fromtimestamp(int(fill['time'])/1000, timezone.utc).isoformat()
                    connection.execute('INSERT INTO cooldowns VALUES (?,?,?,?) ON CONFLICT(namespace,symbol) DO UPDATE SET last_event_at=excluded.last_event_at,operation_id=excluded.operation_id WHERE excluded.last_event_at>cooldowns.last_event_at',
                                       (namespace, symbol, stamp, operation['operation_id']))
            filled = sum((Decimal(r[0]) for r in connection.execute('SELECT qty FROM fills WHERE namespace=? AND operation_id=?', (namespace, operation['operation_id']))), Decimal(0))
            rank = {'FILLED': 4, 'CANCELED': 3, 'EXPIRED': 3, 'REJECTED': 3, 'PARTIALLY_FILLED': 2, 'NEW': 1}
            observations = [json.loads(r[0]) for r in connection.execute('SELECT payload FROM provider_observations WHERE namespace=? AND operation_id=?', (namespace, operation['operation_id']))]
            current = max(observations, key=lambda p: (number(p['executedQty']), rank.get(p['status'], 0), int(p['updateTime'])))
            reported = number(current['executedQty'])
            if min(filled, reported) < 0 or max(filled, reported) > requested:
                raise StateUnavailable('Provider fill exceeds requested quantity')
            cost = sum((Decimal(r[0]) for r in connection.execute('SELECT quote_cost FROM fills WHERE namespace=? AND operation_id=?', (namespace, operation['operation_id']))), Decimal(0))
            status = current['status']
            state = 'fills_pending' if filled != reported or cost != number(current['cummulativeQuoteQty']) else {'FILLED': 'filled', 'CANCELED': 'canceled', 'EXPIRED': 'expired', 'REJECTED': 'rejected', 'NEW': 'accepted', 'PARTIALLY_FILLED': 'partially_filled'}.get(status, 'unknown')
            dispatch = connection.execute("SELECT payload FROM journal WHERE namespace=? AND operation_id=? AND event='exchange_dispatching' ORDER BY sequence LIMIT 1", (namespace, operation['operation_id'])).fetchone()
            if dispatch is None:
                raise StateUnavailable('Durable dispatch reservation unavailable')
            reservation = json.loads(dispatch[0])
            remaining = max(Decimal(0), requested-filled)
            reserved = Decimal(reservation['reservation_amount'])*remaining/requested
            connection.execute('UPDATE reservations SET amount=? WHERE namespace=? AND operation_id=?',
                               (decimal_text(reserved), namespace, operation['operation_id']))
            connection.execute('UPDATE intents SET state=?,provider_order_id=? WHERE namespace=? AND operation_id=?',
                               (state, str(observed['orderId']), namespace, operation['operation_id']))
            if state in {'filled', 'canceled', 'expired', 'rejected'}:
                connection.execute("UPDATE reservations SET state='released' WHERE namespace=? AND operation_id=?", (namespace, operation['operation_id']))
            self._event(connection, 'exchange_order_observed', {'state': state, 'filled': decimal_text(filled)}, operation['operation_id'])
            self.store.record_exchange_risk(connection, now=self.owner.clock())

    @financial_arithmetic
    def process_pending(self) -> int:
        if not self.store.config()[1].get('execution_policy'):
            return 0
        snapshot = self.store.snapshot()
        has_queued = any(operation['state'] == 'queued' for operation in snapshot['operations'])
        recent = self.last_reconcile_attempt is not None and 0 <= self.owner.clock()-self.last_reconcile_attempt < self.owner.policy.reconciliation_interval
        if recent and (not has_queued or snapshot['exchange_account']['state'] != 'ready'):
            try:
                self.protection.maintain()
            except (ProviderFailure, TimeoutError, ConnectionError) as error:
                self._health('reconciliation_required', getattr(error, 'kind', 'transient'))
            return 0
        try:
            self.reconcile()
        except ProviderFailure as error:
            self._health('reconciliation_required', error.kind)
            return 0
        snapshot = self.store.snapshot()
        if snapshot['exchange_account']['state'] != 'ready':
            self.protection.maintain()
            return 0
        count = 0
        with self.store.connection() as connection:
            queued = [dict(r) for r in connection.execute("SELECT * FROM intents WHERE namespace=? AND state='queued' ORDER BY intake_sequence", (self.store.namespace,))]
        for operation in queued:
            if operation['side'] == 'sell':
                try:
                    if not self.protection.prepare_close(operation):
                        continue
                except (ProviderFailure, TimeoutError, ConnectionError) as error:
                    self._health('reconciliation_required', getattr(error, 'kind', 'transient'))
                    return count
            try:
                observed = self._request('entry' if operation['side'] == 'buy' else 'reduction', 1,
                                         lambda prepared: self.provider.submit(prepared[0], operation['side'], prepared[1], operation['provider_client_id']),
                                         prepare=lambda: self.admit(operation))
                self.observe(operation, observed)
            except SubmissionRejected as error:
                with self.store.connection(write=True) as connection:
                    if not self.owner._matches_owner(connection):
                        raise OwnershipUnavailable('Exchange owner uncertain')
                    connection.execute("UPDATE intents SET state='rejected',error_class=? WHERE namespace=? AND operation_id=?",
                                       (error.kind, self.store.namespace, operation['operation_id']))
                    connection.execute("UPDATE reservations SET state='released' WHERE namespace=? AND operation_id=?",
                                       (self.store.namespace, operation['operation_id']))
                    self._event(connection, 'exchange_rejected', {'reason': error.kind, 'authoritative': True}, operation['operation_id'])
                self._health('reconciliation_required', error.kind)
                return count
            except (ConfigurationError, ValueError) as error:
                with self.store.connection(write=True) as connection:
                    if not self.owner._matches_owner(connection):
                        raise OwnershipUnavailable('Exchange owner uncertain')
                    current = connection.execute('SELECT state FROM intents WHERE namespace=? AND operation_id=?',
                                                 (self.store.namespace, operation['operation_id'])).fetchone()
                    if current is not None and current['state'] != 'queued':
                        raise StateUnavailable('Submitted order requires provider reconciliation') from None
                    connection.execute("UPDATE intents SET state='rejected',error_class=? WHERE namespace=? AND operation_id=?",
                                       (str(error)[:128], self.store.namespace, operation['operation_id']))
                    self._event(connection, 'exchange_rejected', {'reason': str(error)[:128]}, operation['operation_id'])
                continue
            except (ProviderFailure, TimeoutError, ConnectionError) as error:
                with self.store.connection(write=True) as connection:
                    if not self.owner._matches_owner(connection):
                        raise OwnershipUnavailable('Exchange owner uncertain')
                    dispatched = connection.execute("SELECT 1 FROM intents WHERE namespace=? AND operation_id=? AND state='dispatching'",
                                                    (self.store.namespace, operation['operation_id'])).fetchone() is not None
                    if dispatched:
                        connection.execute("UPDATE intents SET state='unknown',error_class='dispatch_unknown' WHERE namespace=? AND operation_id=?", (self.store.namespace, operation['operation_id']))
                        self._event(connection, 'exchange_dispatch_unknown', {}, operation['operation_id'])
                self._health('reconciliation_required', 'dispatch_unknown' if dispatched else getattr(error, 'kind', 'transient'))
                return count
            count += 1
            if not self._safe_reconcile():
                return count
            self.protection.maintain()
            if not self._safe_reconcile():
                return count
            if self.store.snapshot()['exchange_account']['state'] != 'ready':
                break
        self.protection.maintain()
        return count

    @financial_arithmetic
    def admit(self, operation: dict[str, Any]) -> tuple[str, str]:
        with self.store.connection(write=True) as connection:
            if not self.owner._matches_owner(connection):
                raise OwnershipUnavailable('Exchange owner uncertain')
            current = connection.execute('SELECT state FROM intents WHERE namespace=? AND operation_id=?',
                                         (self.store.namespace, operation['operation_id'])).fetchone()
            if current is None or current['state'] != 'queued':
                raise StateUnavailable('Order is no longer queued for dispatch')
            version, config = self.store._config(connection)
            policy = config['execution_policy']
            quantity = self.dispatch_quantity(operation)
            if quantity <= 0 or operation['side'] not in {'buy', 'sell'} or operation['kind'] not in {'trade', 'close'}:
                raise ConfigurationError('Canonical positive order required')
            symbol = operation['symbol']
            price_row = connection.execute('SELECT price,fetched_at FROM market_prices WHERE namespace=? AND symbol=?', (self.store.namespace, symbol)).fetchone()
            if price_row is None or operation['config_version'] != version:
                raise ConfigurationError('Fresh reviewed configuration and quote required')
            timestamp = datetime.fromisoformat(price_row['fetched_at'])
            if timestamp.tzinfo is None or not 0 <= self.owner.clock()-timestamp.timestamp() < self.owner.policy.price_expiry:
                raise ConfigurationError('Fresh reviewed quote required')
            price = number(price_row['price'])
            review_row = connection.execute("SELECT payload FROM journal WHERE namespace=? AND operation_id=? AND event='intent_queued'", (self.store.namespace, operation['operation_id'])).fetchone()
            if review_row is None:
                raise ConfigurationError('Reviewed quote required')
            review = json.loads(review_row[0])
            if not isinstance(review.get('reviewed_at'), str):
                raise ConfigurationError('Reviewed quote required')
            reviewed_at = datetime.fromisoformat(review['reviewed_at'])
            if reviewed_at.tzinfo is None or not 0 <= self.owner.clock()-reviewed_at.timestamp() < self.owner.policy.price_expiry or number(review.get('reviewed_price')) != price:
                raise ConfigurationError('Price changed or review expired; review again')
            from market_runtime import observed_health
            scan = self.store._scan(connection)
            health = observed_health({**scan, 'ownership': self.store._ownership(connection),
                                      'config_version': version, 'market_prices': self.store._prices(connection),
                                      'market_budget': self.store._market_budget(connection)}, self.owner.clock())
            if not health['execution_ready'] or symbol not in health.get('coverage', []):
                raise ConfigurationError('Fresh completed market data required')
            try:
                validate_market_quantity(self.store._scan(connection).get('market_metadata', {}), symbol, quantity, price, self.owner.clock(), self.store.venue)
                formatted = self.provider.quantity(symbol, decimal_text(quantity))
            except ValueError as error:
                raise ConfigurationError(str(error)) from error
            if number(formatted) != quantity:
                raise ConfigurationError('Quantity does not match provider precision')
            base, quote = symbol.split('/')
            if quote != policy['quote']:
                raise ConfigurationError('Configured funding quote required')
            amount = quantity*price*(1+Decimal(policy['slippage_fraction']))*(1+Decimal(policy['fee_fraction'])) if operation['side'] == 'buy' else quantity
            asset = quote if operation['side'] == 'buy' else base
            account = self.store._exchange_account(connection)
            if account['state'] != 'ready':
                raise ConfigurationError('Provider reconciliation required')
            if amount > Decimal(account['balances'].get(asset, {}).get('available', '0')):
                raise ConfigurationError('Insufficient confirmed free assets')
            if operation['side'] == 'buy':
                from exchange_protection import protection_terms
                try:
                    protection_terms(scan.get('market_metadata', {}), symbol, quantity, price, config,
                                     self.owner.clock(), self.store.venue)
                except ValueError as error:
                    raise ConfigurationError(str(error)) from error
                risk = self.store.exchange_risk(connection, now=self.owner.clock())
                if risk['entry_blocked']:
                    raise ConfigurationError(risk['reason'])
                positions = self.store._positions(connection)
                if any(p['basis_status'] == 'provider_basis_unknown' for p in positions):
                    raise ConfigurationError('provider_history_review_required')
                cooldown = connection.execute('SELECT last_event_at FROM cooldowns WHERE namespace=? AND symbol=?', (self.store.namespace, symbol)).fetchone()
                if cooldown is not None and self.owner.clock()-datetime.fromisoformat(cooldown[0]).timestamp() < max(policy['cooldown_seconds'], config['cooldown']):
                    raise ConfigurationError('entry_cooldown')
                if symbol not in {p['symbol'] for p in positions} and len(positions) >= min(policy['max_positions'], config['max_positions']):
                    raise ConfigurationError('position_capacity_exceeded')
                prices = self.store._prices(connection)
                exposure = Decimal(0)
                for position in positions:
                    if position['symbol'] not in prices:
                        raise ConfigurationError('Confirmed holding price unavailable')
                    exposure += Decimal(position['qty'])*number(prices[position['symbol']]['price'])
                equity = Decimal(account['balances'].get(quote, {}).get('total', '0')) + exposure
                if amount > equity*Decimal(policy['max_allocation_fraction']):
                    raise ConfigurationError('allocation_cap_exceeded')
                if exposure + quantity*price*(1+Decimal(policy['slippage_fraction'])) > equity*Decimal(policy['max_exposure_fraction']):
                    raise ConfigurationError('exposure_cap_exceeded')
                loss = quantity*price*(1+Decimal(policy['slippage_fraction']))*(Decimal(str(config['stop_loss_pct']))+2*Decimal(policy['fee_fraction'])+Decimal(policy['slippage_fraction']))
                if loss > equity*Decimal(policy['risk_fraction']):
                    raise ConfigurationError('trade_risk_cap_exceeded')
                validate_position_limit(self.store._scan(connection).get('market_metadata', {}), symbol,
                                        Decimal(account['balances'].get(base, {}).get('total', '0')) + quantity)
            connection.execute('INSERT INTO reservations VALUES (?,?,?,?,?)', (self.store.namespace, operation['operation_id'], asset, decimal_text(amount), 'active'))
            connection.execute("UPDATE intents SET state='dispatching' WHERE namespace=? AND operation_id=?", (self.store.namespace, operation['operation_id']))
            connection.execute("UPDATE execution_owners SET state='recovering' WHERE token=? AND generation=?",
                               (self.owner.token, self.owner.generation))
            self._event(connection, 'exchange_health', {'state': 'reconciliation_required', 'reason': 'dispatch_in_flight',
                                                       'as_of': self.owner.clock(), 'owner_generation': self.owner.generation})
            self._event(connection, 'exchange_dispatching', {'client_order_id': operation['provider_client_id'],
                                                            'reservation_amount': decimal_text(amount), 'asset': asset}, operation['operation_id'])
        return symbol, formatted
