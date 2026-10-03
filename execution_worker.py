"""Owned, atomic legacy paper effects. Capital/protection policy belongs to BT05."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Any
from collections.abc import Callable
import time
from state_store import StateStore
from operations_policy import OperationsPolicy

from execution_owner import ExecutionOwner, OwnershipUnavailable


class ExecutionWorker(ExecutionOwner):
    def __init__(self, store: StateStore, *, clock: Callable[[], float] = time.time) -> None:
        super().__init__(store)
        self.clock = clock
        self.policy = OperationsPolicy()
    def publish_scan(self, snapshot: dict[str, Any]) -> None:
        if not self.ownership_confirmed():
            raise OwnershipUnavailable('Execution ownership uncertain')
        allowed = {'last_scan', 'universe', 'live_prices', 'signals', 'exit_signals', 'logs', 'performance', 'cooldowns', 'market_health', 'market_cache'}
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
                               (self.store.namespace, version, 'legacy-dashboard-v1', json.dumps(values, allow_nan=False), now))
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

    def process_pending(self) -> int:
        if not self.ownership_confirmed():
            raise OwnershipUnavailable('Execution ownership uncertain')
        # No provider dispatch is implemented in this foundation. Exchange
        # recovery and fills require the separately approved BT06 lifecycle.
        if self.store.mode != 'paper':
            return 0
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
            operations = connection.execute("SELECT * FROM intents WHERE namespace=? AND state IN ('queued','waiting_for_market_data') ORDER BY intake_sequence",
                                            (self.store.namespace,)).fetchall()
            for operation in operations:
                # Paper projection, simulated fill and journal commit together;
                # a crash/failed commit leaves the original queued intent intact.
                connection.execute('SAVEPOINT paper_operation')
                try:
                    self._paper_effect(connection, operation)
                except (ValueError, InvalidOperation) as error:
                    connection.execute('ROLLBACK TO paper_operation')
                    connection.execute('RELEASE paper_operation')
                    connection.execute("UPDATE intents SET state='rejected',error_class=? WHERE namespace=? AND operation_id=?",
                                       (type(error).__name__, self.store.namespace, operation['operation_id']))
                    connection.execute('INSERT INTO journal(namespace,operation_id,event,payload) VALUES (?,?,?,?)',
                                       (self.store.namespace, operation['operation_id'], 'paper_rejected', json.dumps({'error_class': type(error).__name__})))
                else:
                    connection.execute('RELEASE paper_operation')
                processed += 1
        return processed

    def _paper_effect(self, connection: sqlite3.Connection, operation: sqlite3.Row) -> None:
        namespace, symbol = self.store.namespace, operation['symbol']
        requested = Decimal(operation['requested_qty'])
        # Projection preconditions: invalid data cannot become a confirmed row.
        # BT05 still owns request validation, capital/risk/fees and credible fills.
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
        price = Decimal(price_row[0])
        if not price.is_finite() or price <= 0:
            raise ValueError('Invalid paper projection price')
        position = connection.execute('SELECT qty,entry FROM positions WHERE namespace=? AND symbol=?', (namespace, symbol)).fetchone()
        held = Decimal(position['qty']) if position else Decimal(0)
        entry = Decimal(position['entry']) if position else price
        quantity = requested
        if operation['side'] == 'buy':
            resulting = held + quantity
            entry = (held * entry + quantity * price) / resulting
            connection.execute('''INSERT INTO positions VALUES (?,?,?,?,?)
                                  ON CONFLICT(namespace,symbol) DO UPDATE SET qty=excluded.qty,entry=excluded.entry''',
                               (namespace, symbol, str(resulting), str(entry), 'legacy-paper-profile; fees/capital incomplete'))
        else:
            if held <= 0:
                raise ValueError('No paper position')
            quantity = min(requested, held)
            remaining = held - quantity
            if remaining > 0:
                connection.execute('UPDATE positions SET qty=? WHERE namespace=? AND symbol=?', (str(remaining), namespace, symbol))
            else:
                connection.execute('DELETE FROM positions WHERE namespace=? AND symbol=?', (namespace, symbol))
        connection.execute('''INSERT INTO fills(namespace,fill_id,operation_id,symbol,side,qty,price,quote_cost,profile)
                              VALUES (?,?,?,?,?,?,?,?,?)''',
                           (namespace, operation['operation_id'] + ':paper:1', operation['operation_id'], symbol,
                            operation['side'], str(quantity), str(price), str(quantity * price), 'legacy-paper-v1'))
        connection.execute("UPDATE intents SET state='filled',error_class=NULL WHERE namespace=? AND operation_id=?", (namespace, operation['operation_id']))
        connection.execute('INSERT INTO journal(namespace,operation_id,event,payload) VALUES (?,?,?,?)',
                           (namespace, operation['operation_id'], 'paper_fill', json.dumps({'qty': str(quantity), 'price': str(price), 'profile': 'legacy-paper-v1'})))
