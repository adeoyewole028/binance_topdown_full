"""Durable protection policy; simulated fills stay in the owned paper executor."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
import uuid
from typing import Any, TYPE_CHECKING

from paper_policy import decimal_text, financial_arithmetic, number
from state_store import StateUnavailable
from configuration import ConfigurationError

if TYPE_CHECKING:
    from execution_worker import ExecutionWorker


@financial_arithmetic
def exit_price(stop: Decimal, target: Decimal, *, opened: Decimal, high: Decimal,
               low: Decimal) -> tuple[str, Decimal] | None:
    """Stop wins an ambiguous bar; adverse stop gaps and conservative TP fills."""
    if min(stop, target, opened, high, low) <= 0 or high < max(opened, low) or low > opened or stop >= target:
        raise ValueError('Invalid protection prices')
    if low <= stop:
        return 'stop_loss', min(stop, opened)
    if high >= target:
        return 'take_profit', target
    return None


class PaperProtection:
    def __init__(self, owner: ExecutionWorker) -> None:
        self.owner, self.store = owner, owner.store

    def event(self, connection: Any, group: str, values: dict[str, Any]) -> None:
        connection.execute('INSERT INTO journal(namespace,operation_id,event,payload) VALUES (?,?,?,?)',
                           (self.store.namespace, group, 'protection_observed', json.dumps(values, sort_keys=True, allow_nan=False)))

    @financial_arithmetic
    def sync(self, connection: Any) -> None:
        namespace = self.store.namespace
        positions = {p['symbol']: p for p in self.store._positions(connection)}
        groups = self.store._protection(connection)
        active = {g['symbol']: g for g in groups if g['state'] not in {'closed', 'canceled'}}
        rules = self.store._position_rules(connection)
        for symbol in set(active) | set(positions):
            position, old = positions.get(symbol), active.get(symbol)
            if old and position and Decimal(old['quantity']) == Decimal(position['qty']):
                continue
            if old:
                connection.execute("UPDATE protection_groups SET state='closed' WHERE namespace=? AND group_id=?", (namespace, old['group_id']))
                self.event(connection, old['group_id'], {'exit_reason': 'holding_changed', 'semantics': 'simulation'})
            if position is None:
                continue
            rule = rules[symbol]
            entry = Decimal(position['entry'])
            group = uuid.uuid4().hex
            entry_operation = connection.execute("SELECT operation_id FROM fills WHERE namespace=? AND symbol=? AND side='buy' ORDER BY rowid DESC LIMIT 1", (namespace, symbol)).fetchone()
            if entry_operation is None:
                # Legacy imports have no entry to attach an invented policy to.
                continue
            bound = rule['state'] == 'bound'
            values: dict[str, Any] = {'semantics': 'simulation', 'config_version': rule.get('config_version'),
                                      'installed_at': self.owner.clock(), 'exit_reason': None,
                                      'next_action': None if bound else 'Review entry history; protection policy cannot be inferred'}
            if bound:
                values.update(stop_price=decimal_text(entry*(1-Decimal(str(rule['config']['stop_loss_pct'])))),
                              take_profit_price=decimal_text(entry*(1+Decimal(str(rule['config']['take_profit_pct'])))))
            state = 'active' if bound else 'failed'
            connection.execute('INSERT INTO protection_groups VALUES (?,?,?,?,?,?,?,?)',
                               (namespace, group, symbol, entry_operation[0], position['qty'], state, None,
                                datetime.fromtimestamp(self.owner.clock(), timezone.utc).isoformat()))
            self.event(connection, group, values)

    def trigger(self, connection: Any) -> None:
        """Fresh quotes trigger owned simulated exits before new entries."""
        for group in self.store._protection(connection):
            if group['state'] != 'active':
                continue
            row = connection.execute('SELECT price,fetched_at FROM market_prices WHERE namespace=? AND symbol=?', (self.store.namespace, group['symbol'])).fetchone()
            if row is None:
                continue
            stamp = datetime.fromisoformat(row['fetched_at'])
            if stamp.tzinfo is None or not 0 <= self.owner.clock()-stamp.timestamp() < self.owner.policy.price_expiry:
                continue
            price = number(row['price'])
            stop, target = Decimal(group['stop_price']), Decimal(group['take_profit_price'])
            hit = None
            scan = self.store._scan(connection)
            cache = scan.get('market_cache', {}).get(f'{self.store.venue}:{group["symbol"]}:15m')
            if cache:
                from market_runtime import validated_bars
                server_time = self.owner.clock()+float(scan.get('market_health', {}).get('clock_skew_seconds', 0))
                for bar in validated_bars(cache['rows'], '15m', server_time):
                    # A candle containing the entry has unknown intrabar order.
                    # Only a whole candle after installation is eligible.
                    if bar[0]/1000 >= group['installed_at']:
                        hit = exit_price(stop, target, opened=Decimal(str(bar[1])), high=Decimal(str(bar[2])), low=Decimal(str(bar[3])))
                        if hit:
                            break
            if hit is None:
                hit = exit_price(stop, target, opened=price, high=price, low=price)
            if hit is None:
                continue
            reason, fill_price = hit
            # Queued paper sells have no remote effect; release them before the
            # serial protection exit reserves the confirmed remaining holding.
            pending = connection.execute("SELECT operation_id FROM intents WHERE namespace=? AND symbol=? AND side='sell' AND state IN ('queued','waiting_for_setup','waiting_for_market_data')", (self.store.namespace, group['symbol'])).fetchall()
            for operation in pending:
                connection.execute("UPDATE intents SET state='canceled',error_class='superseded_by_protection' WHERE namespace=? AND operation_id=?", (self.store.namespace, operation[0]))
                connection.execute("UPDATE reservations SET state='released' WHERE namespace=? AND operation_id=?", (self.store.namespace, operation[0]))
            operation = uuid.uuid4().hex
            version = self.store._config(connection)[0]
            connection.execute('''INSERT INTO intents(namespace,operation_id,idempotency_key,request_hash,kind,symbol,side,requested_qty,config_version,provider_client_id)
                                  VALUES (?,?,?,?,?,?,?,?,?,?)''',
                               (self.store.namespace, operation, operation, 'paper-protection:'+group['group_id'], 'close', group['symbol'], 'sell', group['quantity'], version, operation))
            self.event(connection, group['group_id'], {'exit_reason': reason, 'semantics': 'simulation', 'exit_operation': operation})
            connection.execute('SAVEPOINT protective_exit')
            try:
                row = connection.execute('SELECT * FROM intents WHERE namespace=? AND operation_id=?', (self.store.namespace, operation)).fetchone()
                self.owner._paper_effect(connection, row, protective_price=fill_price)
                state = connection.execute('SELECT state FROM intents WHERE namespace=? AND operation_id=?', (self.store.namespace, operation)).fetchone()[0]
                if state != 'filled':
                    raise ValueError('Fresh completed market data required for simulated protection')
            except (ValueError, ConfigurationError) as error:
                connection.execute('ROLLBACK TO protective_exit')
                connection.execute("UPDATE intents SET state='rejected',error_class='protection_failed' WHERE namespace=? AND operation_id=?", (self.store.namespace, operation))
                connection.execute("UPDATE protection_groups SET state='failed' WHERE namespace=? AND group_id=?", (self.store.namespace, group['group_id']))
                self.event(connection, group['group_id'], {'next_action': 'Review unprotected simulated holding and rejected exit', 'reason': str(error)[:128]})
            finally:
                connection.execute('RELEASE protective_exit')
            current = connection.execute('SELECT qty FROM positions WHERE namespace=? AND symbol=?', (self.store.namespace, group['symbol'])).fetchone()
            if current is None:
                connection.execute("UPDATE protection_groups SET state='closed' WHERE namespace=? AND group_id=?", (self.store.namespace, group['group_id']))
            elif Decimal(current[0]) < 0:
                raise StateUnavailable('Protection oversold holdings')
