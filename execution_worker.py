"""Owned atomic paper accounting and explicit exchange/protection coordination."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING
from types import TracebackType
from collections.abc import Callable
import time
from state_store import StateStore
from operations_policy import OperationsPolicy
from configuration import ConfigurationError
from paper_policy import financial_arithmetic

from execution_owner import ExecutionOwner, OwnershipUnavailable

if TYPE_CHECKING:
    from automation import Automation


class ExecutionWorker(ExecutionOwner):
    def __init__(self, store: StateStore, *, clock: Callable[[], float] = time.time, exchange_provider: Any = None) -> None:
        super().__init__(store)
        self.clock = clock
        self.policy = OperationsPolicy()
        self.automation: Automation | None = None
        self.exchange_execution = None
        from position_protection import PaperProtection
        self.paper_protection = PaperProtection(self)
        if exchange_provider is not None:
            if store.mode != 'exchange':
                raise ConfigurationError('Private provider cannot attach to paper execution')
            from exchange_execution import ExchangeExecution
            self.exchange_execution = ExchangeExecution(self, exchange_provider)

    def __exit__(self, error_type: type[BaseException] | None,
                 error: BaseException | None, traceback: TracebackType | None) -> None:
        try:
            if self.automation is not None:
                self.automation.stop()
        finally:
            super().__exit__(error_type, error, traceback)

    def reconcile_exchange(self) -> None:
        if self.exchange_execution is None:
            raise ConfigurationError('Explicit exchange provider required')
        self.exchange_execution.reconcile()

    def poll_exchange(self) -> None:
        if self.exchange_execution is not None:
            self.exchange_execution.reconcile_if_due()
    def publish_scan(self, snapshot: dict[str, Any]) -> None:
        if not self.ownership_confirmed():
            raise OwnershipUnavailable('Execution ownership uncertain')
        allowed = {'last_scan', 'universe', 'live_prices', 'signals', 'exit_signals', 'logs', 'performance', 'cooldowns', 'market_health', 'market_cache', 'market_metadata', 'strategy_decisions', 'strategy_profile', 'strategy_version'}
        values = {key: value for key, value in snapshot.items() if key in allowed}
        now = datetime.fromtimestamp(self.clock(), timezone.utc).isoformat()
        with self.store.connection(write=True) as connection:
            if not self._matches_owner(connection):
                raise OwnershipUnavailable('Execution ownership uncertain')
            version = snapshot.get('config_version')
            if version is None:
                version = connection.execute('SELECT max(version) FROM config_versions WHERE namespace=?', (self.store.namespace,)).fetchone()[0]
            connection.execute('''INSERT INTO scan_snapshots VALUES (?,?,?,?,?) ON CONFLICT(namespace)
                                  DO UPDATE SET config_version=excluded.config_version,profile=excluded.profile,
                                  values_json=excluded.values_json,published_at=excluded.published_at''',
                               (self.store.namespace, version, snapshot.get('strategy_profile','legacy-dashboard-v1'), json.dumps(values, allow_nan=False), now))
            unresolved = connection.execute('''SELECT 1 FROM intents i JOIN namespaces n ON n.id=i.namespace
                WHERE n.account=? AND n.venue=? AND i.state IN ('dispatching','unknown','canceling','processing') LIMIT 1''',
                                            (self.store.account, self.store.venue)).fetchone()
            if self.store.mode == 'paper' and unresolved is None and values.get('market_health', {}).get('state') == 'ready':
                connection.execute("UPDATE execution_owners SET state='ready',updated_at=? WHERE token=?", (now, self.token))

    def publish_prices(self, prices: dict[str, Any], *, fetched_at: float | None = None) -> None:
        parsed_prices = {symbol: Decimal(str(price)) for symbol, price in prices.items()}
        if not parsed_prices or any(not p.is_finite() or p <= 0 for p in parsed_prices.values()):
            raise ValueError('Invalid public prices')
        received = self.clock() if fetched_at is None else fetched_at
        if not math.isfinite(received) or not 0 <= received <= self.clock():
            raise ValueError('Invalid public price receipt time')
        now = datetime.fromtimestamp(received, timezone.utc).isoformat()
        with self.store.connection(write=True) as connection:
            if not self._matches_owner(connection):
                raise OwnershipUnavailable('Execution ownership uncertain')
            for symbol, parsed in parsed_prices.items():
                connection.execute('''INSERT INTO market_prices VALUES (?,?,?,?) ON CONFLICT(namespace,symbol)
                                      DO UPDATE SET price=excluded.price,fetched_at=excluded.fetched_at''',
                                   (self.store.namespace, symbol, str(parsed), now))
            if self.store.mode == 'paper':
                self.store.record_paper_risk(connection)
            else:
                self.store.record_exchange_risk(connection, now=self.clock())

    def process_pending(self) -> int:
        if not self.ownership_confirmed():
            raise OwnershipUnavailable('Execution ownership uncertain')
        # Private submission exists only on the explicitly attached owned adapter.
        if self.store.mode != 'paper':
            return self.exchange_execution.process_pending() if self.exchange_execution is not None else 0
        self.store.positions()  # Corrupt required projections block all dispatch.
        processed = 0
        with self.store.connection(write=True) as connection:
            if not self._matches_owner(connection):
                raise OwnershipUnavailable('Execution ownership uncertain')
            unresolved = connection.execute('''SELECT 1 FROM intents i JOIN namespaces n ON n.id=i.namespace
                                               WHERE n.account=? AND n.venue=? AND i.state IN ('dispatching','unknown','canceling','processing') LIMIT 1''',
                                            (self.store.account, self.store.venue)).fetchone()
            if unresolved is not None:
                return 0
            connection.execute("UPDATE execution_owners SET state='ready' WHERE token=?", (self.token,))
            self.paper_protection.sync(connection)
            self.paper_protection.trigger(connection)
            operations = connection.execute("SELECT * FROM intents WHERE namespace=? AND state IN ('queued','waiting_for_market_data','waiting_for_setup') ORDER BY intake_sequence",
                                            (self.store.namespace,)).fetchall()
            for operation in operations:
                # Paper projection, simulated fill and journal commit together;
                # a crash/failed commit leaves the original queued intent intact.
                connection.execute('SAVEPOINT paper_operation')
                try:
                    self._paper_effect(connection, operation)
                except (ValueError, InvalidOperation, ConfigurationError) as error:
                    connection.execute('ROLLBACK TO paper_operation')
                    connection.execute('RELEASE paper_operation')
                    connection.execute("UPDATE intents SET state='rejected',error_class=? WHERE namespace=? AND operation_id=?",
                                       (str(error)[:128], self.store.namespace, operation['operation_id']))
                    connection.execute("UPDATE reservations SET state='released' WHERE namespace=? AND operation_id=?", (self.store.namespace, operation['operation_id']))
                    connection.execute('INSERT INTO journal(namespace,operation_id,event,payload) VALUES (?,?,?,?)',
                                       (self.store.namespace, operation['operation_id'], 'paper_rejected', json.dumps({'error_class': str(error)[:128]})))
                else:
                    connection.execute('RELEASE paper_operation')
                processed += 1
                self.paper_protection.sync(connection)
        return processed

    @financial_arithmetic
    def _paper_effect(self, connection: sqlite3.Connection, operation: sqlite3.Row, *, protective_price: Decimal | None = None) -> None:
        namespace, symbol = self.store.namespace, operation['symbol']
        policy = self.store._config(connection)[1].get('paper_policy')
        if not policy:
            connection.execute("UPDATE intents SET state='waiting_for_setup' WHERE namespace=? AND operation_id=?", (namespace, operation['operation_id']))
            return
        requested = Decimal(operation['requested_qty'])
        # Invalid requests cannot become financial projections.
        if not requested.is_finite() or requested <= 0 or operation['side'] not in {'buy', 'sell'}:
            raise ValueError('Invalid paper projection input')
        price_row = connection.execute('SELECT price,fetched_at FROM market_prices WHERE namespace=? AND symbol=?', (namespace, symbol)).fetchone()
        fresh = False
        if price_row is not None:
            try:
                timestamp = datetime.fromisoformat(price_row['fetched_at'])
                fresh = timestamp.tzinfo is not None and 0 <= self.clock() - timestamp.timestamp() < self.policy.price_expiry
            except (ValueError, TypeError, OverflowError):
                pass
        from market_runtime import observed_health
        scan = self.store._scan(connection)
        health = observed_health({**scan, 'ownership': self.store._ownership(connection),
                                  'config_version': self.store._config(connection)[0],
                                  'market_prices': self.store._prices(connection),
                                  'market_budget': self.store._market_budget(connection)}, self.clock())
        if not fresh or not health['execution_ready'] or symbol not in health.get('coverage', []):
            connection.execute("UPDATE intents SET state='waiting_for_market_data' WHERE namespace=? AND operation_id=?",
                               (namespace, operation['operation_id']))
            return
        assert price_row is not None
        review = connection.execute("SELECT payload FROM journal WHERE namespace=? AND operation_id=? AND event='intent_queued'", (namespace, operation['operation_id'])).fetchone()
        if protective_price is None and (review is None or operation['config_version'] != self.store._config(connection)[0]):
            raise ValueError('configuration_changed_review_required')
        if protective_price is None:
            assert review is not None
            reviewed_context = json.loads(review[0])
            reviewed_stamp = datetime.fromisoformat(reviewed_context['reviewed_at'])
            if reviewed_stamp.tzinfo is None or not 0 <= self.clock() - reviewed_stamp.timestamp() < self.policy.price_expiry:
                raise ValueError('reviewed_price_expired')
            reviewed_price = reviewed_context.get('reviewed_price')
            if reviewed_price is None or Decimal(reviewed_price) != Decimal(price_row[0]):
                raise ValueError('price_changed_review_required')
        price = Decimal(price_row[0])
        if not price.is_finite() or price <= 0:
            raise ValueError('Invalid paper projection price')
        from paper_filters import validate_market_quantity
        validate_market_quantity(scan.get('market_metadata', {}), symbol, requested, price, self.clock(), self.store.venue)
        self.store.reserve_paper(connection, operation['operation_id'], symbol, operation['side'], requested, price, now=self.clock())
        if protective_price is not None:
            if operation['side'] != 'sell' or protective_price <= 0:
                raise ValueError('Invalid protective reduction')
            price = protective_price
        price *= 1 + Decimal(policy['slippage_fraction']) * (1 if operation['side'] == 'buy' else -1)
        cost = requested * price
        fee = cost * Decimal(policy['fee_fraction'])
        base, quote = symbol.split('/')
        if quote != policy['quote']:
            raise ValueError('Symbol quote differs from paper funding asset')
        balances = self.store._paper_account(connection, exclude_operation=operation['operation_id'])['balances']
        if operation['side'] == 'buy' and cost + fee > Decimal(balances[quote]['available']):
            raise ValueError('Insufficient paper capital')
        position = connection.execute('SELECT qty,entry FROM positions WHERE namespace=? AND symbol=?', (namespace, symbol)).fetchone()
        held = Decimal(position['qty']) if position else Decimal(0)
        entry = Decimal(position['entry']) if position else price
        quantity = requested
        if operation['side'] == 'buy':
            resulting = held + quantity
            entry = (held * entry + cost + fee) / resulting
            connection.execute('''INSERT INTO positions VALUES (?,?,?,?,?)
                                  ON CONFLICT(namespace,symbol) DO UPDATE SET qty=excluded.qty,entry=excluded.entry''',
                               (namespace, symbol, str(resulting), str(entry), policy['model']))
        else:
            if held < requested or requested > Decimal(balances.get(base, {}).get('available', '0')):
                raise ValueError('Insufficient paper holdings')
            quantity = requested
            remaining = held - quantity
            if remaining > 0:
                connection.execute('UPDATE positions SET qty=? WHERE namespace=? AND symbol=?', (str(remaining), namespace, symbol))
            else:
                connection.execute('DELETE FROM positions WHERE namespace=? AND symbol=?', (namespace, symbol))
        connection.execute('''INSERT INTO fills(namespace,fill_id,operation_id,symbol,side,qty,price,quote_cost,fee_amount,fee_asset,profile)
                              VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                           (namespace, operation['operation_id'] + ':paper:1', operation['operation_id'], symbol,
                            operation['side'], str(quantity), str(price), str(cost), str(fee), quote, policy['model']))
        if operation['side'] == 'buy':
            connection.execute('INSERT INTO cooldowns VALUES (?,?,?,?) ON CONFLICT(namespace,symbol) DO UPDATE SET last_event_at=excluded.last_event_at,operation_id=excluded.operation_id',
                               (namespace, symbol, datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(), operation['operation_id']))
        direction = Decimal(1) if operation['side'] == 'buy' else Decimal(-1)
        for asset, movement in ((base, direction * quantity), (quote, -cost - fee if direction == 1 else cost - fee)):
            connection.execute('INSERT INTO balance_movements(namespace,movement_id,asset,amount,event) VALUES (?,?,?,?,?)',
                               (namespace, operation['operation_id'] + ':' + asset, asset, str(movement), 'paper_fill'))
        connection.execute("UPDATE reservations SET state='released' WHERE namespace=? AND operation_id=?", (namespace, operation['operation_id']))
        connection.execute("UPDATE intents SET state='filled',error_class=NULL WHERE namespace=? AND operation_id=?", (namespace, operation['operation_id']))
        connection.execute('INSERT INTO journal(namespace,operation_id,event,payload) VALUES (?,?,?,?)',
                           (namespace, operation['operation_id'], 'paper_fill', json.dumps({'qty': str(quantity), 'price': str(price), 'profile': policy['model'], 'fee_amount': str(fee), 'fee_asset': quote})))
        self.store.record_paper_risk(connection)
