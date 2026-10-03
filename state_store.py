"""Transactional, namespace-qualified state. Opening never creates missing state."""
from __future__ import annotations

from contextlib import closing, contextmanager
import json
from pathlib import Path
import sqlite3
import math
import re
import time
from typing import Any, Iterator
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from configuration import validated_config, ConfigurationError, ConfigurationConflict
from paper_policy import decimal_text, financial_arithmetic


class StateUnavailable(RuntimeError):
    """Required durable state is unavailable; financial writes must stop."""


class StateStore:
    def __init__(self, directory: Path, account: str, venue: str, mode: str) -> None:
        self.path = directory / 'trading.sqlite3'
        self.marker = directory / 'state-identity.json'
        self.namespace = str(uuid.uuid5(uuid.NAMESPACE_URL, f'binance-topdown:{account}:{venue}:{mode}'))
        self.account, self.venue, self.mode = account, venue, mode

    @contextmanager
    def connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        try:
            identity = json.loads(self.marker.read_text(encoding='utf-8'))
            with closing(sqlite3.connect(self.path.as_uri() + '?mode=rw', uri=True,
                                         timeout=5, isolation_level=None)) as connection:
                connection.row_factory = sqlite3.Row
                connection.execute('PRAGMA foreign_keys=ON')
                connection.execute('PRAGMA synchronous=FULL')
                row = connection.execute('SELECT id,schema_version FROM store_identity').fetchone()
                if row is None or row[0] != identity['id'] or row[1] != 1 or identity['schema_version'] != 1:
                    raise StateUnavailable('Durable state identity mismatch')
                if write:
                    connection.execute('BEGIN IMMEDIATE')
                try:
                    yield connection
                    if write:
                        connection.commit()
                except BaseException:
                    if write:
                        connection.rollback()
                    raise
        except (OSError, sqlite3.Error, ValueError, KeyError, InvalidOperation) as error:
            raise StateUnavailable('Required durable state unavailable') from error

    def provision(self, defaults: dict[str, Any]) -> None:
        """Explicit initialization only. Never overwrite or heal existing state."""
        if self.path.exists() or self.marker.exists():
            with self.connection(write=True) as connection:
                existing = connection.execute('SELECT id FROM namespaces WHERE id=?', (self.namespace,)).fetchone()
                if existing is None:
                    connection.execute('INSERT INTO namespaces VALUES (?,?,?,?)', (self.namespace, self.account, self.venue, self.mode))
                    values = validated_config({**defaults, 'paper_trading': self.mode == 'paper'})
                    connection.execute('INSERT INTO config_versions(namespace,version,values_json,actor) VALUES (?,1,?,?)',
                                       (self.namespace, json.dumps(values, sort_keys=True, allow_nan=False), 'explicit-provision'))
            self.provision_market()
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        identity = str(uuid.uuid4())
        with closing(sqlite3.connect(self.path, isolation_level=None)) as connection:
            connection.executescript('''
                PRAGMA foreign_keys=ON;
                PRAGMA synchronous=FULL;
                BEGIN IMMEDIATE;
                CREATE TABLE store_identity (id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL);
                CREATE TABLE namespaces (
                    id TEXT PRIMARY KEY, account TEXT NOT NULL, venue TEXT NOT NULL CHECK(venue IN ('testnet','mainnet')),
                    mode TEXT NOT NULL CHECK(mode IN ('paper','exchange')), UNIQUE(account,venue,mode));
                CREATE TABLE config_versions (
                    namespace TEXT NOT NULL REFERENCES namespaces(id), version INTEGER NOT NULL,
                    values_json TEXT NOT NULL, actor TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(namespace,version));
                CREATE TRIGGER config_immutable_update BEFORE UPDATE ON config_versions
                    BEGIN SELECT RAISE(ABORT,'Immutable configuration'); END;
                CREATE TRIGGER config_immutable_delete BEFORE DELETE ON config_versions
                    BEGIN SELECT RAISE(ABORT,'Immutable configuration'); END;
                CREATE TABLE intents (
                    namespace TEXT NOT NULL REFERENCES namespaces(id), operation_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL, request_hash TEXT NOT NULL, kind TEXT NOT NULL,
                    symbol TEXT NOT NULL, side TEXT NOT NULL, requested_qty TEXT NOT NULL,
                    config_version INTEGER NOT NULL, state TEXT NOT NULL DEFAULT 'queued',
                    provider_client_id TEXT NOT NULL, provider_order_id TEXT, error_class TEXT,
                    intake_sequence INTEGER UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(namespace,operation_id), UNIQUE(namespace,idempotency_key),
                    FOREIGN KEY(namespace,config_version) REFERENCES config_versions(namespace,version));
                CREATE INDEX pending_intents ON intents(namespace,state,intake_sequence);
                CREATE UNIQUE INDEX provider_order_identity ON intents(namespace,provider_order_id)
                    WHERE provider_order_id IS NOT NULL;
                CREATE TABLE journal (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, namespace TEXT NOT NULL REFERENCES namespaces(id),
                    operation_id TEXT, event TEXT NOT NULL, payload TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                CREATE TRIGGER journal_immutable_update BEFORE UPDATE ON journal
                    BEGIN SELECT RAISE(ABORT,'Immutable journal'); END;
                CREATE TRIGGER journal_immutable_delete BEFORE DELETE ON journal
                    BEGIN SELECT RAISE(ABORT,'Immutable journal'); END;
                CREATE TABLE execution_owners (
                    account TEXT NOT NULL, venue TEXT NOT NULL, token TEXT NOT NULL,
                    generation INTEGER NOT NULL, state TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(account,venue));
                CREATE TABLE market_prices (
                    namespace TEXT NOT NULL REFERENCES namespaces(id), symbol TEXT NOT NULL,
                    price TEXT NOT NULL, fetched_at TEXT NOT NULL, PRIMARY KEY(namespace,symbol));
                CREATE TABLE positions (
                    namespace TEXT NOT NULL REFERENCES namespaces(id), symbol TEXT NOT NULL,
                    qty TEXT NOT NULL, entry TEXT NOT NULL, basis_status TEXT NOT NULL,
                    PRIMARY KEY(namespace,symbol));
                CREATE TABLE fills (
                    namespace TEXT NOT NULL, fill_id TEXT NOT NULL, operation_id TEXT NOT NULL,
                    symbol TEXT NOT NULL, side TEXT NOT NULL, qty TEXT NOT NULL, price TEXT NOT NULL,
                    quote_cost TEXT NOT NULL, fee_amount TEXT, fee_asset TEXT, profile TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(namespace,fill_id),
                    FOREIGN KEY(namespace,operation_id) REFERENCES intents(namespace,operation_id));
                CREATE TRIGGER fills_immutable_update BEFORE UPDATE ON fills
                    BEGIN SELECT RAISE(ABORT,'Immutable fill'); END;
                CREATE TRIGGER fills_immutable_delete BEFORE DELETE ON fills
                    BEGIN SELECT RAISE(ABORT,'Immutable fill'); END;
                CREATE TABLE migration_sources (
                    namespace TEXT NOT NULL REFERENCES namespaces(id), checksum TEXT NOT NULL,
                    source_name TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL,
                    record_count INTEGER NOT NULL, imported_count INTEGER NOT NULL DEFAULT 0,
                    quarantine_count INTEGER NOT NULL, contents TEXT NOT NULL,
                    PRIMARY KEY(namespace,checksum));
                CREATE TABLE scan_snapshots (
                    namespace TEXT PRIMARY KEY REFERENCES namespaces(id), config_version INTEGER NOT NULL,
                    profile TEXT NOT NULL, values_json TEXT NOT NULL, published_at TEXT NOT NULL,
                    FOREIGN KEY(namespace,config_version) REFERENCES config_versions(namespace,version));
                CREATE TABLE provider_observations (
                    namespace TEXT NOT NULL, observation_id TEXT NOT NULL, operation_id TEXT NOT NULL,
                    status TEXT NOT NULL, payload TEXT NOT NULL, observed_at TEXT NOT NULL,
                    PRIMARY KEY(namespace,observation_id),
                    FOREIGN KEY(namespace,operation_id) REFERENCES intents(namespace,operation_id));
                CREATE TRIGGER observations_immutable_update BEFORE UPDATE ON provider_observations
                    BEGIN SELECT RAISE(ABORT,'Immutable observation'); END;
                CREATE TRIGGER observations_immutable_delete BEFORE DELETE ON provider_observations
                    BEGIN SELECT RAISE(ABORT,'Immutable observation'); END;
                CREATE TABLE protection_groups (
                    namespace TEXT NOT NULL, group_id TEXT NOT NULL, symbol TEXT NOT NULL,
                    operation_id TEXT NOT NULL, quantity TEXT NOT NULL, state TEXT NOT NULL,
                    provider_list_id TEXT, verified_at TEXT, PRIMARY KEY(namespace,group_id),
                    FOREIGN KEY(namespace,operation_id) REFERENCES intents(namespace,operation_id));
                CREATE UNIQUE INDEX provider_protection_identity ON protection_groups(namespace,provider_list_id)
                    WHERE provider_list_id IS NOT NULL;
                CREATE TABLE balance_movements (
                    namespace TEXT NOT NULL REFERENCES namespaces(id), movement_id TEXT NOT NULL,
                    asset TEXT NOT NULL, amount TEXT NOT NULL, event TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(namespace,movement_id));
                CREATE TRIGGER movements_immutable_update BEFORE UPDATE ON balance_movements
                    BEGIN SELECT RAISE(ABORT,'Immutable balance movement'); END;
                CREATE TRIGGER movements_immutable_delete BEFORE DELETE ON balance_movements
                    BEGIN SELECT RAISE(ABORT,'Immutable balance movement'); END;
                CREATE TABLE reservations (
                    namespace TEXT NOT NULL, operation_id TEXT NOT NULL, asset TEXT NOT NULL,
                    amount TEXT NOT NULL, state TEXT NOT NULL, PRIMARY KEY(namespace,operation_id,asset),
                    FOREIGN KEY(namespace,operation_id) REFERENCES intents(namespace,operation_id));
                CREATE TABLE cooldowns (
                    namespace TEXT NOT NULL REFERENCES namespaces(id), symbol TEXT NOT NULL,
                    last_event_at TEXT NOT NULL, operation_id TEXT NOT NULL, PRIMARY KEY(namespace,symbol),
                    FOREIGN KEY(namespace,operation_id) REFERENCES intents(namespace,operation_id));
            ''')
            connection.execute('INSERT INTO store_identity VALUES (?,1)', (identity,))
            connection.execute('INSERT INTO namespaces VALUES (?,?,?,?)', (self.namespace, self.account, self.venue, self.mode))
            values = validated_config({**defaults, 'paper_trading': self.mode == 'paper'})
            connection.execute('INSERT INTO config_versions(namespace,version,values_json,actor) VALUES (?,1,?,?)',
                               (self.namespace, json.dumps(values, sort_keys=True, allow_nan=False), 'explicit-provision'))
            connection.commit()
        # If the marker write fails, the database remains non-tradable until
        # reviewed repair; startup never silently provisions over that partial state.
        self.marker.write_text(json.dumps({'id': identity, 'schema_version': 1}), encoding='utf-8')
        self.provision_market()

    def provision_market(self) -> None:
        """Explicit additive BT04 extension; existing financial schema stays at v1."""
        self.validate()
        with self.connection(write=True) as connection:
            connection.execute('''CREATE TABLE IF NOT EXISTS market_budget_v1 (
                venue TEXT PRIMARY KEY, values_json TEXT NOT NULL)''')
            connection.execute('INSERT OR IGNORE INTO market_budget_v1 VALUES (?,?)',
                               (self.venue, json.dumps({'limits': [], 'blocked_until': 0, 'failures': 0})))

    def config(self) -> tuple[int, dict[str, Any]]:
        with self.connection() as connection:
            return self._config(connection)

    def _config(self, connection: sqlite3.Connection) -> tuple[int, dict[str, Any]]:
        row = connection.execute('SELECT version,values_json FROM config_versions WHERE namespace=? ORDER BY version DESC LIMIT 1',
                                 (self.namespace,)).fetchone()
        if row is None:
            raise StateUnavailable('Namespace requires explicit initialization')
        values = json.loads(row['values_json'])
        if not isinstance(values, dict) or values.get('paper_trading') is not (self.mode == 'paper'):
            raise StateUnavailable('Invalid durable configuration')
        try:
            values = validated_config(values)
        except ConfigurationError as error:
            raise StateUnavailable('Invalid durable configuration') from error
        return int(row['version']), values

    def update_config(self, changes: dict[str, Any], actor: str, *, expected_version: int | None = None) -> tuple[int, dict[str, Any]]:
        if 'paper_policy' in changes:
            raise ConfigurationError('Use explicit paper setup for financial assumptions')
        with self.connection(write=True) as connection:
            current, values = self._config(connection)
            if expected_version is not None and expected_version != current:
                raise ConfigurationConflict('Configuration changed; review the latest version')
            values.update(changes)
            values['paper_trading'] = self.mode == 'paper'
            values = validated_config(values)
            version = current + 1
            connection.execute('INSERT INTO config_versions(namespace,version,values_json,actor) VALUES (?,?,?,?)',
                               (self.namespace, version, json.dumps(values, sort_keys=True, allow_nan=False), actor))
            connection.execute('INSERT INTO journal(namespace,event,payload) VALUES (?,?,?)',
                               (self.namespace, 'config_updated', json.dumps({'version': version, 'actor': actor})))
            return version, values

    def validate(self) -> None:
        with self.connection() as connection:
            if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok' or connection.execute('PRAGMA foreign_key_check').fetchone():
                raise StateUnavailable('Durable state integrity failure')
            self._market_budget(connection)
            self._prices(connection)
            self._paper_account(connection)
            self._exchange_account(connection)
            self._entry_control(connection)
            self._fills(connection)
            self._protection(connection)
        self.config()
        self.positions()

    @financial_arithmetic
    def enqueue(self, kind: str, symbol: str, side: str, quantity: str, *, idempotency_key: str | None = None, now: float | None = None, sizing: str = 'quantity', review: dict[str, Any] | None = None, automatic_signal: dict[str, Any] | None = None, expected_owner: tuple[str,int] | None = None) -> str:
        import hashlib
        from paper_policy import number
        if kind not in {'trade', 'close'} or side not in {'buy', 'sell'} or not isinstance(symbol, str) or not re.fullmatch(r'[A-Z0-9]{2,12}/[A-Z0-9]{2,12}', symbol):
            raise ConfigurationError('Invalid order side or canonical symbol')
        if sizing not in {'quantity', 'fixed', 'risk', 'full_close'} or (sizing in {'fixed', 'risk'} and side != 'buy') or (sizing == 'full_close' and (kind != 'close' or side != 'sell')):
            raise ConfigurationError('Invalid sizing mode')
        amount = number(quantity) if sizing == 'quantity' else Decimal(0)
        if sizing == 'quantity' and amount <= 0:
            raise ConfigurationError('Order quantity must be positive')
        quantity = decimal_text(amount) if sizing == 'quantity' else ''
        operation = uuid.uuid4().hex
        key = operation if idempotency_key is None else idempotency_key
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', key):
            raise ConfigurationError('Invalid idempotency key')
        request = json.dumps({'kind': kind, 'symbol': symbol, 'side': side, 'qty': quantity, 'sizing': sizing}, sort_keys=True, allow_nan=False)
        with self.connection(write=True) as connection:
            if automatic_signal is not None:
                owner = connection.execute('SELECT token,generation,state FROM execution_owners WHERE account=? AND venue=?', (self.account,self.venue)).fetchone()
                if expected_owner is None or owner is None or (owner['token'],owner['generation']) != expected_owner or owner['state'] != 'ready':
                    raise ConfigurationConflict('Automation ownership/recovery required')
            digest = hashlib.sha256(request.encode()).hexdigest()
            repeated = connection.execute('SELECT operation_id,request_hash FROM intents WHERE namespace=? AND idempotency_key=?', (self.namespace, key)).fetchone()
            if repeated is not None:
                if repeated['request_hash'] != digest:
                    raise ConfigurationConflict('Idempotency key already used for a different order')
                return str(repeated['operation_id'])
            row = connection.execute('SELECT max(version) FROM config_versions WHERE namespace=?',
                                     (self.namespace,)).fetchone()
            if row is None or row[0] is None:
                raise StateUnavailable('Namespace requires explicit initialization')
            reviewed = connection.execute('SELECT price,fetched_at FROM market_prices WHERE namespace=? AND symbol=?', (self.namespace, symbol)).fetchone()
            if review is not None:
                current_quote = dict(reviewed) if reviewed else None
                if review.get('namespace') != self.namespace or review.get('config_version') != row[0] or review.get('prices', {}).get(symbol) != current_quote:
                    raise ConfigurationConflict('Price or settings changed; refresh the order review')
            elif self._config(connection)[1].get('paper_policy'):
                raise ConfigurationConflict('Reviewed order context required')
            if sizing == 'full_close':
                held = connection.execute('SELECT qty FROM positions WHERE namespace=? AND symbol=?', (self.namespace, symbol)).fetchone()
                if held is None:
                    raise ConfigurationError('No confirmed holding to close')
                if review is not None and review.get('holdings', {}).get(symbol) != held[0]:
                    raise ConfigurationConflict('Holding changed; refresh the close review')
                amount = number(held[0])
                quantity = decimal_text(amount)
            elif sizing != 'quantity':
                amount = self.size_paper(connection, symbol, sizing) if self.mode == 'paper' else self._size_order(connection, symbol, sizing, exchange=True)
                quantity = decimal_text(amount)
            connection.execute('''INSERT INTO intents(namespace,operation_id,idempotency_key,request_hash,kind,symbol,side,
                                   requested_qty,config_version,provider_client_id) VALUES (?,?,?,?,?,?,?,?,?,?)''',
                               (self.namespace, operation, key, digest,
                                kind, symbol, side, quantity, row[0], operation))
            context = {**json.loads(request), 'reviewed_price': reviewed['price'] if reviewed else None,
                       'reviewed_at': reviewed['fetched_at'] if reviewed else None, 'config_version': row[0],
                       'strategy_profile':self._config(connection)[1]['strategy_profile'], 'strategy_version':1}
            if automatic_signal is not None:
                context.update(source='automation', signal_id=automatic_signal['signal_id'],
                    signal_profile=automatic_signal['profile'],signal_profile_version=automatic_signal['profile_version'],
                    signal_config_version=automatic_signal['config_version'],execution_close=automatic_signal['execution_close'])
            event = connection.execute('INSERT INTO journal(namespace,operation_id,event,payload) VALUES (?,?,?,?)',
                                       (self.namespace, operation, 'intent_queued', json.dumps(context)))
            connection.execute('UPDATE intents SET intake_sequence=? WHERE namespace=? AND operation_id=?',
                               (event.lastrowid, self.namespace, operation))
            if self.mode == 'paper' and self._config(connection)[1].get('paper_policy'):
                try:
                    if reviewed is None:
                        raise ValueError('reviewed_price_unavailable')
                    stamp = datetime.fromisoformat(reviewed['fetched_at'])
                    from operations_policy import OperationsPolicy
                    current_time = time.time() if now is None else now
                    if stamp.tzinfo is None or not 0 <= current_time - stamp.timestamp() < OperationsPolicy().price_expiry:
                        raise ValueError('reviewed_price_expired')
                    self.reserve_paper(connection, operation, symbol, side, amount, Decimal(reviewed['price']), now=current_time)
                except ValueError as error:
                    connection.execute("UPDATE intents SET state='rejected',error_class=? WHERE namespace=? AND operation_id=?",
                                       (str(error), self.namespace, operation))
                    connection.execute('INSERT INTO journal(namespace,operation_id,event,payload) VALUES (?,?,?,?)',
                                       (self.namespace, operation, 'paper_rejected', json.dumps({'reason': str(error)})))
        return operation

    @financial_arithmetic
    def size_paper(self, connection: sqlite3.Connection, symbol: str, mode: str) -> Decimal:
        return self._size_order(connection, symbol, mode, exchange=False)

    @financial_arithmetic
    def _size_order(self, connection: sqlite3.Connection, symbol: str, mode: str, *, exchange: bool) -> Decimal:
        from decimal import ROUND_FLOOR
        config = self._config(connection)[1]
        policy = config.get('execution_policy' if exchange else 'paper_policy')
        prices = self._prices(connection)
        if not policy or symbol not in prices or symbol.split('/')[1] != policy['quote']:
            raise ConfigurationError('Explicit setup and reviewed symbol price required for sizing')
        account = self._exchange_account(connection) if exchange else self._paper_account(connection)
        if exchange and account['state'] != 'ready':
            raise ConfigurationError('Provider reconciliation required for sizing')
        exposure = sum((Decimal(p['qty']) * Decimal(prices[p['symbol']]['price']) for p in self._positions(connection)), Decimal(0))
        equity = Decimal(account['balances'][policy['quote']]['total']) + exposure
        if exchange:
            risk = self.exchange_risk(connection, now=time.time())
            if risk.get('equity') is None or risk['entry_blocked']:
                raise ConfigurationError('Confirmed capital and risk valuation required for sizing')
            equity = Decimal(risk['equity'])
        price = Decimal(prices[symbol]['price']) * (1 + Decimal(policy['slippage_fraction']))
        unit_cost = price * (1 + Decimal(policy['fee_fraction']))
        budget = min(Decimal(account['balances'][policy['quote']]['available']),
                     equity * Decimal(policy['max_allocation_fraction']),
                     equity * Decimal(policy['max_exposure_fraction']) - exposure)
        if mode == 'fixed':
            budget = min(budget, Decimal(policy['fixed_quote']))
        quantity = budget / unit_cost
        if mode == 'risk':
            loss = price * (Decimal(str(config['stop_loss_pct'])) + 2 * Decimal(policy['fee_fraction']) + Decimal(policy['slippage_fraction']))
            quantity = min(quantity, equity * min(Decimal(policy['risk_fraction']), Decimal(str(config['risk']))) / loss)
        metadata = self._scan(connection).get('market_metadata', {})
        try:
            market = next(s for s in metadata['symbols'] if s['baseAsset'] + '/' + s['quoteAsset'] == symbol)
            steps = [Decimal(f['stepSize']) for f in market['filters'] if f['filterType'] in {'LOT_SIZE', 'MARKET_LOT_SIZE'} and Decimal(f['stepSize']) > 0]
            if not steps:
                raise ValueError('No quantity precision')
            step = max(steps)
            quantity = (quantity / step).to_integral_value(rounding=ROUND_FLOOR) * step
        except (KeyError, StopIteration, ValueError) as error:
            raise ConfigurationError('Verified quantity rules required for sizing') from error
        if quantity <= 0:
            raise ConfigurationError('No available capital for a valid order')
        return quantity

    @financial_arithmetic
    def reserve_paper(self, connection: sqlite3.Connection, operation: str, symbol: str, side: str,
                      quantity: Decimal, observed_price: Decimal, *, now: float | None = None) -> None:
        policy = self._config(connection)[1]['paper_policy']
        base, quote = symbol.split('/')
        if quote != policy['quote']:
            raise ValueError('paper_quote_mismatch')
        if not observed_price.is_finite() or observed_price <= 0:
            raise ValueError('invalid_price')
        price = observed_price * (1 + Decimal(policy['slippage_fraction']) * (1 if side == 'buy' else -1))
        amount = quantity * price * (1 + Decimal(policy['fee_fraction'])) if side == 'buy' else quantity
        asset = quote if side == 'buy' else base
        account = self._paper_account(connection, exclude_operation=operation)
        if amount > Decimal(account['balances'].get(asset, {}).get('available', '0')):
            raise ValueError('insufficient_paper_assets')
        if side == 'buy':
            if account['entry_blocked']:
                raise ValueError(account['entry_block_reason'])
            from paper_filters import validate_position_limit
            pending_base = sum((Decimal(row[0]) for row in connection.execute(
                "SELECT i.requested_qty FROM reservations r JOIN intents i ON i.namespace=r.namespace AND i.operation_id=r.operation_id WHERE r.namespace=? AND r.state='active' AND i.side='buy' AND i.symbol LIKE ? AND i.operation_id!=?",
                (self.namespace, base + '/%', operation))), Decimal(0))
            validate_position_limit(self._scan(connection).get('market_metadata', {}), symbol,
                                    Decimal(account['balances'].get(base, {}).get('total', '0')) + pending_base + quantity)
            now = time.time() if now is None else now
            cooldown = connection.execute('SELECT last_event_at FROM cooldowns WHERE namespace=? AND symbol=?', (self.namespace, symbol)).fetchone()
            if cooldown is not None and now - datetime.fromisoformat(cooldown[0]).timestamp() < max(policy['cooldown_seconds'], self._config(connection)[1]['cooldown']):
                raise ValueError('entry_cooldown')
            positions = self._positions(connection)
            slots = {p['symbol'] for p in positions}
            slots.update(row[0] for row in connection.execute("SELECT i.symbol FROM reservations r JOIN intents i ON i.namespace=r.namespace AND i.operation_id=r.operation_id WHERE r.namespace=? AND r.state='active' AND i.side='buy' AND i.operation_id!=?", (self.namespace, operation)))
            if symbol not in slots and len(slots) >= min(policy['max_positions'], self._config(connection)[1]['max_positions']):
                raise ValueError('position_capacity_exceeded')
            prices = self._prices(connection)
            exposure = Decimal(0)
            for position in positions:
                if position['basis_status'] != policy['model'] or position['symbol'].split('/')[1] != quote:
                    raise ValueError('paper_history_review_required')
                if position['symbol'] not in prices:
                    raise ValueError('held_price_unavailable')
                exposure += Decimal(position['qty']) * Decimal(prices[position['symbol']]['price'])
            equity = Decimal(account['balances'][quote]['total']) + exposure
            pending = sum((Decimal(row[0]) for row in connection.execute("SELECT r.amount FROM reservations r JOIN intents i ON i.namespace=r.namespace AND i.operation_id=r.operation_id WHERE r.namespace=? AND r.state='active' AND i.side='buy' AND r.operation_id!=?", (self.namespace, operation))), Decimal(0))
            if amount > equity * Decimal(policy['max_allocation_fraction']):
                raise ValueError('allocation_cap_exceeded')
            if exposure + pending + quantity * price > equity * Decimal(policy['max_exposure_fraction']):
                raise ValueError('exposure_cap_exceeded')
            stop = Decimal(str(self._config(connection)[1]['stop_loss_pct']))
            loss = quantity * price * (stop + 2 * Decimal(policy['fee_fraction']) + Decimal(policy['slippage_fraction']))
            if loss > equity * Decimal(policy['risk_fraction']):
                raise ValueError('trade_risk_cap_exceeded')
        connection.execute('INSERT INTO reservations VALUES (?,?,?,?,?) ON CONFLICT(namespace,operation_id,asset) DO UPDATE SET amount=excluded.amount,state=excluded.state',
                           (self.namespace, operation, asset, decimal_text(amount), 'active'))

    def operations(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            return self._operations(connection)

    def _operations(self, connection: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = [dict(row) for row in connection.execute('''SELECT i.operation_id,i.kind,i.symbol,i.side,i.requested_qty,
                       i.config_version,i.state,i.error_class,c.values_json FROM intents i
                       JOIN config_versions c ON c.namespace=i.namespace AND c.version=i.config_version
                       WHERE i.namespace=? AND i.kind!='protection' ORDER BY i.intake_sequence''', (self.namespace,))]
        for row in rows:
            values = validated_config(json.loads(row.pop('values_json')))
            row.update(strategy_profile=values['strategy_profile'],strategy_version=values['strategy_version'],source='manual')
            event = connection.execute("SELECT payload FROM journal WHERE namespace=? AND operation_id=? AND event='intent_queued' ORDER BY sequence LIMIT 1", (self.namespace,row['operation_id'])).fetchone()
            if event:
                context = json.loads(event[0])
                row.update({key:context[key] for key in ('source','signal_id','signal_profile','signal_profile_version','signal_config_version','execution_close') if key in context})
        return rows

    def set_price(self, symbol: str, price: str) -> None:
        with self.connection(write=True) as connection:
            connection.execute('''INSERT INTO market_prices VALUES (?,?,?,?)
                                  ON CONFLICT(namespace,symbol) DO UPDATE SET price=excluded.price,fetched_at=excluded.fetched_at''',
                               (self.namespace, symbol, price, datetime.now(timezone.utc).isoformat()))

    def positions(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            return self._positions(connection)

    def prices(self) -> dict[str, Any]:
        with self.connection() as connection:
            return self._prices(connection)

    def market_coverage(self) -> set[str]:
        with self.connection() as connection:
            connection.execute('BEGIN')
            return {row[0] for row in connection.execute('''SELECT symbol FROM positions WHERE namespace=?
                UNION SELECT symbol FROM intents WHERE namespace=? AND state NOT IN ('filled','rejected','canceled')
                UNION SELECT symbol FROM protection_groups WHERE namespace=? AND state NOT IN ('closed','canceled','rejected','failed')''',
                                                        (self.namespace,) * 3)}

    def _prices(self, connection: sqlite3.Connection) -> dict[str, Any]:
        values = {row['symbol']: {**dict(row), 'venue': self.venue} for row in connection.execute(
            'SELECT symbol,price,fetched_at FROM market_prices WHERE namespace=?', (self.namespace,))}
        try:
            for quote in values.values():
                price = Decimal(quote['price'])
                if not price.is_finite() or price <= 0:
                    raise StateUnavailable('Invalid durable public price')
        except InvalidOperation as error:
            raise StateUnavailable('Invalid durable public price') from error
        return values

    def _market_budget(self, connection: sqlite3.Connection) -> dict[str, Any]:
        if connection.execute("SELECT 1 FROM sqlite_master WHERE name='market_budget_v1' AND type='table'").fetchone() is None:
            return {'state': 'uninitialized', 'reason': 'market_budget_unavailable'}
        row = connection.execute('SELECT values_json FROM market_budget_v1 WHERE venue=?', (self.venue,)).fetchone()
        if row is None:
            return {'state': 'uninitialized', 'reason': 'market_budget_unavailable'}
        values = json.loads(row[0])
        if (not isinstance(values, dict) or not isinstance(values.get('limits'), list)
                or type(values.get('blocked_until')) not in (int, float) or not math.isfinite(values['blocked_until'])):
            raise StateUnavailable('Invalid durable market budget')
        for limit in values['limits']:
            if not isinstance(limit, dict) or limit['kind'] not in {'REQUEST_WEIGHT', 'RAW_REQUESTS'} or any(
                type(limit[key]) is not int or limit[key] < (0 if key in {'used', 'start'} else 1)
                for key in ('duration', 'limit', 'used', 'start')):
                raise StateUnavailable('Invalid durable market budget')
        return {'state': 'auth_blocked' if values.get('auth_blocked') else 'configured' if values['limits'] else 'uninitialized',
                'limits': values['limits'], 'blocked_until': values['blocked_until'], 'reason': values.get('last_failure')}

    def _positions(self, connection: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = [dict(row) for row in connection.execute('SELECT symbol,qty,entry,basis_status FROM positions WHERE namespace=? ORDER BY symbol',
                                                      (self.namespace,))]
        try:
            for row in rows:
                for field in ('qty', 'entry'):
                    value = Decimal(row[field])
                    unknown_entry = field == 'entry' and row['basis_status'] == 'provider_basis_unknown' and value == 0
                    if not value.is_finite() or (value <= 0 and not unknown_entry):
                        raise StateUnavailable('Invalid durable position')
                    row[field] = decimal_text(value)
        except InvalidOperation as error:
            raise StateUnavailable('Invalid durable position') from error
        return rows

    def position_rules(self) -> dict[str, Any]:
        with self.connection() as connection:
            connection.execute('BEGIN')
            return self._position_rules(connection)

    def _position_rules(self, connection: sqlite3.Connection) -> dict[str, Any]:
        result = {}
        for position in self._positions(connection):
            quantity = Decimal(0)
            versions: set[int] = set()
            for fill in connection.execute('''SELECT f.qty,f.side,f.fee_amount,f.fee_asset,i.config_version FROM fills f
                JOIN intents i ON i.namespace=f.namespace AND i.operation_id=f.operation_id
                WHERE f.namespace=? AND f.symbol=? ORDER BY f.rowid''', (self.namespace, position['symbol'])):
                try:
                    amount = Decimal(fill['qty'])
                except InvalidOperation as error:
                    raise StateUnavailable('Invalid durable fill') from error
                if not amount.is_finite() or amount <= 0 or fill['side'] not in {'buy', 'sell'}:
                    raise StateUnavailable('Invalid durable fill')
                if fill['side'] == 'buy':
                    versions.add(int(fill['config_version']))
                    quantity += amount
                else:
                    quantity -= amount
                if fill['fee_asset'] == position['symbol'].split('/')[0]:
                    fee = Decimal(fill['fee_amount'])
                    if not fee.is_finite() or fee < 0:
                        raise StateUnavailable('Invalid durable base fee')
                    quantity -= fee
                if quantity == 0:
                    versions.clear()
            # No FIFO/LIFO/lot allocation is invented for mixed entry versions.
            rule: dict[str, Any] = {'state': 'review_required', 'config_version': None}
            if (len(versions) == 1 and quantity == Decimal(position['qty'])
                    and not position['basis_status'].startswith('legacy-opening')):
                version = next(iter(versions))
                row = connection.execute('SELECT values_json FROM config_versions WHERE namespace=? AND version=?',
                                         (self.namespace, version)).fetchone()
                if row is not None:
                    try:
                        values = validated_config(json.loads(row[0]))
                    except ConfigurationError:
                        pass
                    else:
                        rule.update(state='bound', config_version=version, config=values)
            result[position['symbol']] = rule
        return result

    def migration(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            return self._migration(connection)

    def _migration(self, connection: sqlite3.Connection) -> list[dict[str, Any]]:
        return [dict(row) for row in connection.execute('''SELECT checksum,source_name,kind,state,
                record_count,imported_count,quarantine_count FROM migration_sources
                WHERE namespace=? ORDER BY source_name,checksum''', (self.namespace,))]

    def ownership(self) -> dict[str, Any]:
        with self.connection() as connection:
            return self._ownership(connection)

    def _ownership(self, connection: sqlite3.Connection) -> dict[str, Any]:
        row = connection.execute('SELECT generation,state,updated_at FROM execution_owners WHERE account=? AND venue=?',
                                 (self.account, self.venue)).fetchone()
        return dict(row) if row is not None else {'generation': 0, 'state': 'unowned'}

    def scan(self) -> dict[str, Any]:
        with self.connection() as connection:
            return self._scan(connection)

    def _scan(self, connection: sqlite3.Connection) -> dict[str, Any]:
        row = connection.execute('SELECT values_json FROM scan_snapshots WHERE namespace=?', (self.namespace,)).fetchone()
        if row is None:
            return {}
        values = json.loads(row[0])
        if not isinstance(values, dict):
            raise StateUnavailable('Invalid durable scan')
        return values

    def setup_paper(self, values: dict[str, Any], actor: str, *, expected_version: int) -> None:
        from paper_policy import validated_policy
        if self.mode != 'paper':
            raise ConfigurationError('Paper setup cannot fund exchange state')
        policy = validated_policy(values)
        with self.connection(write=True) as connection:
            version, config = self._config(connection)
            if version != expected_version:
                raise ConfigurationConflict('Configuration changed; review before paper setup')
            previous = config.get('paper_policy')
            if previous and any(previous[k] != policy[k] for k in ('quote', 'initial_cash')):
                raise ConfigurationConflict('Paper seed and quote are immutable; use a new isolated account')
            if not previous:
                connection.execute('INSERT INTO balance_movements VALUES (?,?,?,?,?,?)',
                                   (self.namespace, 'paper-seed', policy['quote'], policy['initial_cash'],
                                    'paper_seed', datetime.now(timezone.utc).isoformat()))
            config['paper_policy'] = policy
            config['cooldown'] = policy['cooldown_seconds']
            connection.execute('INSERT INTO config_versions(namespace,version,values_json,actor) VALUES (?,?,?,?)',
                               (self.namespace, version + 1, json.dumps(config, sort_keys=True), actor))
            connection.execute('INSERT INTO journal(namespace,event,payload) VALUES (?,?,?)',
                               (self.namespace, 'paper_setup', json.dumps({'version': version + 1, 'actor': actor, 'model': policy['model']})))

    def setup_execution(self, values: dict[str, Any], actor: str, *, expected_version: int) -> None:
        from paper_policy import validated_execution_policy
        if self.mode != 'exchange':
            raise ConfigurationError('Exchange policy requires an exchange namespace')
        policy = validated_execution_policy(values)
        with self.connection(write=True) as connection:
            version, config = self._config(connection)
            if version != expected_version:
                raise ConfigurationConflict('Configuration changed; review exchange policy')
            if config.get('execution_policy', {}).get('quote', policy['quote']) != policy['quote']:
                raise ConfigurationConflict('Funding quote requires an isolated reviewed namespace')
            config['execution_policy'] = policy
            config['cooldown'] = policy['cooldown_seconds']
            connection.execute('INSERT INTO config_versions(namespace,version,values_json,actor) VALUES (?,?,?,?)',
                               (self.namespace, version+1, json.dumps(config,sort_keys=True), actor))
            connection.execute('INSERT INTO journal(namespace,event,payload) VALUES (?,?,?)',
                               (self.namespace, 'exchange_policy_setup', json.dumps({'version': version+1,'actor': actor})))

    def cancel_paper(self, operation: str, actor: str) -> None:
        if self.mode != 'paper':
            raise ConfigurationError('Paper cancellation cannot affect exchange orders')
        with self.connection(write=True) as connection:
            row = connection.execute('SELECT state FROM intents WHERE namespace=? AND operation_id=?', (self.namespace, operation)).fetchone()
            if row is None:
                raise ConfigurationError('Unknown paper operation')
            if row[0] == 'canceled':
                return
            if row[0] not in {'queued', 'waiting_for_setup', 'waiting_for_market_data'}:
                raise ConfigurationConflict('Operation is terminal or executing; cancellation cannot undo a fill')
            connection.execute("UPDATE intents SET state='canceled' WHERE namespace=? AND operation_id=?", (self.namespace, operation))
            connection.execute("UPDATE reservations SET state='released' WHERE namespace=? AND operation_id=?", (self.namespace, operation))
            connection.execute('INSERT INTO journal(namespace,operation_id,event,payload) VALUES (?,?,?,?)',
                               (self.namespace, operation, 'paper_canceled', json.dumps({'actor': actor})))

    def _fills(self, connection: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = [dict(row) for row in connection.execute('SELECT fill_id,operation_id,symbol,side,qty,price,quote_cost,fee_amount,fee_asset,profile,created_at FROM fills WHERE namespace=? ORDER BY rowid', (self.namespace,))]
        for row in rows:
            for field in ('qty', 'price', 'quote_cost', 'fee_amount'):
                if row[field] is not None:
                    value = Decimal(row[field])
                    if not value.is_finite() or value < 0 or (field in {'qty', 'price'} and value == 0):
                        raise StateUnavailable('Invalid confirmed fill')
                    row[field] = decimal_text(value)
        return rows

    @financial_arithmetic
    def _paper_account(self, connection: sqlite3.Connection, *, exclude_operation: str = "") -> dict[str, Any]:
        if self.mode != 'paper':
            return {'state': 'not_applicable', 'balances': {}}
        policy = self._config(connection)[1].get('paper_policy')
        totals: dict[str, Decimal] = {}
        for row in connection.execute('SELECT asset,amount FROM balance_movements WHERE namespace=?', (self.namespace,)):
            amount = Decimal(row['amount'])
            if not amount.is_finite():
                raise StateUnavailable('Invalid paper balance movement')
            totals[row['asset']] = totals.get(row['asset'], Decimal(0)) + amount
        reserved: dict[str, Decimal] = {}
        for row in connection.execute("SELECT asset,amount FROM reservations WHERE namespace=? AND state='active' AND operation_id!=?", (self.namespace, exclude_operation)):
            reserved[row['asset']] = reserved.get(row['asset'], Decimal(0)) + Decimal(row['amount'])
        for asset, total in totals.items():
            if not total.is_finite() or total < 0 or not reserved.get(asset, Decimal(0)).is_finite() or not 0 <= reserved.get(asset, Decimal(0)) <= total:
                raise StateUnavailable('Invalid paper balances or reservations')
        equity = Decimal(totals.get(policy['quote'], Decimal(0))) if policy else Decimal(0)
        incomplete = False
        for position in self._positions(connection):
            price = connection.execute('SELECT price FROM market_prices WHERE namespace=? AND symbol=?', (self.namespace, position['symbol'])).fetchone()
            if not policy or position['basis_status'] != policy['model'] or price is None or position['symbol'].split('/')[1] != policy['quote']:
                incomplete = True
            else:
                equity += Decimal(position['qty']) * Decimal(price[0])
        peak = Decimal(policy['initial_cash']) if policy else Decimal(0)
        for row in connection.execute("SELECT payload FROM journal WHERE namespace=? AND event='paper_risk_checkpoint'", (self.namespace,)):
            peak = max(peak, Decimal(json.loads(row[0])['equity']))
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak if peak else Decimal(0)
        halted = connection.execute("SELECT 1 FROM journal WHERE namespace=? AND event='paper_entry_stop' LIMIT 1", (self.namespace,)).fetchone() is not None
        protection_failed = connection.execute("SELECT 1 FROM protection_groups WHERE namespace=? AND state IN ('failed','unknown','installing','canceling') LIMIT 1", (self.namespace,)).fetchone() is not None
        reason = 'paper_history_review_required' if incomplete else 'drawdown_limit' if policy and drawdown >= Decimal(policy['drawdown_fraction']) else 'entry_halted' if halted else 'protection_review_required' if protection_failed else None
        return {'state': 'configured' if policy else 'setup_required', 'policy': policy,
                'entry_blocked': bool(reason) or not policy, 'entry_block_reason': reason or ('setup_required' if not policy else None),
                'equity': None if incomplete or not policy else decimal_text(equity),
                'peak_equity': None if incomplete or not policy else decimal_text(peak),
                'drawdown': None if incomplete or not policy else decimal_text(drawdown),
                'balances': {asset: {'total': decimal_text(total), 'reserved': decimal_text(reserved.get(asset, Decimal(0))),
                                     'available': decimal_text(total - reserved.get(asset, Decimal(0)))} for asset, total in totals.items()}}

    @financial_arithmetic
    def record_paper_risk(self, connection: sqlite3.Connection) -> None:
        account = self._paper_account(connection)
        if account.get('equity') is None:
            return
        connection.execute('INSERT INTO journal(namespace,event,payload) VALUES (?,?,?)',
                           (self.namespace, 'paper_risk_checkpoint', json.dumps({'equity': account['equity']})))
        if account.get('entry_block_reason') == 'drawdown_limit':
            connection.execute('INSERT INTO journal(namespace,event,payload) VALUES (?,?,?)',
                               (self.namespace, 'paper_entry_stop', json.dumps({'reason': 'drawdown_limit'})))

    @financial_arithmetic
    def _exchange_account(self, connection: sqlite3.Connection, *, exclude_operation: str = '') -> dict[str, Any]:
        if self.mode != 'exchange':
            return {'state': 'not_applicable', 'balances': {}}
        row = connection.execute("SELECT payload FROM journal WHERE namespace=? AND event='exchange_account_observed' ORDER BY sequence DESC LIMIT 1", (self.namespace,)).fetchone()
        health = connection.execute("SELECT payload FROM journal WHERE namespace=? AND event='exchange_health' ORDER BY sequence DESC LIMIT 1", (self.namespace,)).fetchone()
        review = connection.execute("SELECT payload FROM journal WHERE namespace=? AND event='exchange_review_required' ORDER BY sequence DESC LIMIT 1", (self.namespace,)).fetchone()
        if row is None:
            return {'state': 'reconciliation_required', **(json.loads(health[0]) if health else {}),
                    **(json.loads(review[0]) if review else {}), 'balances': {}}
        observed = json.loads(row[0])
        balances = {}
        assets = set(observed['balances'])
        assets.update(r[0] for r in connection.execute('SELECT DISTINCT asset FROM balance_movements WHERE namespace=?', (self.namespace,)))
        for asset in assets:
            values = observed['balances'].get(asset, {'free': '0', 'locked': '0'})
            free, locked = Decimal(values['free']), Decimal(values['locked'])
            movement = sum((Decimal(r[0]) for r in connection.execute('SELECT amount FROM balance_movements WHERE namespace=? AND asset=? AND rowid>?', (self.namespace, asset, observed['movement_sequence']))), Decimal(0))
            total = free + locked + movement
            reserved = sum((Decimal(r[0]) for r in connection.execute("SELECT amount FROM reservations WHERE namespace=? AND asset=? AND state='active' AND operation_id!=?", (self.namespace, asset, exclude_operation))), Decimal(0))
            if not total.is_finite() or not locked.is_finite() or not reserved.is_finite() or min(total, locked, reserved) < 0:
                raise StateUnavailable('Invalid exchange accounting')
            balances[asset] = {'total': decimal_text(total), 'locked': decimal_text(locked),
                               'reserved': decimal_text(reserved), 'available': decimal_text(max(Decimal(0), total-max(locked,reserved)))}
        return {**observed, **(json.loads(health[0]) if health else {'state': 'reconciliation_required'}),
                **(json.loads(review[0]) if review else {}), 'balances': balances}

    @financial_arithmetic
    def exchange_risk(self, connection: sqlite3.Connection, *, now: float) -> dict[str, Any]:
        policy = self._config(connection)[1].get('execution_policy')
        if not policy:
            return {'entry_blocked': True, 'reason': 'setup_required', 'equity': None}
        account = self._exchange_account(connection)
        if not account.get('identity'):
            return {'entry_blocked': True, 'reason': 'reconciliation_required', 'equity': None}
        equity = Decimal(account['balances'].get(policy['quote'], {}).get('total', '0'))
        for asset, balance in account['balances'].items():
            if asset == policy['quote'] or Decimal(balance['total']) == 0:
                continue
            row = connection.execute('SELECT price,fetched_at FROM market_prices WHERE namespace=? AND symbol=?', (self.namespace, asset+'/'+policy['quote'])).fetchone()
            if row is None:
                return {'entry_blocked': True, 'reason': 'valuation_unavailable', 'equity': None}
            stamp = datetime.fromisoformat(row['fetched_at'])
            from operations_policy import OperationsPolicy
            if stamp.tzinfo is None or not 0 <= now-stamp.timestamp() < OperationsPolicy().price_expiry:
                return {'entry_blocked': True, 'reason': 'valuation_stale', 'equity': None}
            equity += Decimal(balance['total'])*Decimal(row['price'])
        peak = equity
        for row in connection.execute("SELECT payload FROM journal WHERE namespace=? AND event='exchange_risk_checkpoint'", (self.namespace,)):
            value = Decimal(json.loads(row[0])['equity'])
            if not value.is_finite() or value < 0:
                raise StateUnavailable('Invalid exchange risk history')
            peak = max(peak, value)
        halted = connection.execute("SELECT 1 FROM journal WHERE namespace=? AND event='exchange_entry_stop' LIMIT 1", (self.namespace,)).fetchone() is not None
        drawdown = (peak-equity)/peak if peak else Decimal(0)
        protection_failed = connection.execute("SELECT 1 FROM protection_groups WHERE namespace=? AND state IN ('failed','unknown','installing','canceling') LIMIT 1", (self.namespace,)).fetchone() is not None
        for position in self._positions(connection):
            if position['basis_status'] == 'provider_basis_unknown':
                continue
            covered = sum((Decimal(row[0]) for row in connection.execute("SELECT quantity FROM protection_groups WHERE namespace=? AND symbol=? AND state='active'", (self.namespace, position['symbol']))), Decimal(0))
            if covered != Decimal(position['qty']):
                protection_failed = True
        reason = 'entry_halted' if halted else 'drawdown_limit' if drawdown >= Decimal(policy['drawdown_fraction']) else (account.get('reason') or 'reconciliation_required') if account['state'] != 'ready' else 'protection_review_required' if protection_failed else 'entry_paused' if self._entry_control(connection)['paused'] else 'unfunded_account' if equity <= 0 else None
        return {'entry_blocked': reason is not None, 'reason': reason, 'equity': decimal_text(equity),
                'peak_equity': decimal_text(peak), 'drawdown': decimal_text(drawdown)}

    def record_exchange_risk(self, connection: sqlite3.Connection, *, now: float) -> None:
        risk = self.exchange_risk(connection, now=now)
        if risk.get('equity') is None:
            return
        connection.execute('INSERT INTO journal(namespace,event,payload) VALUES (?,?,?)',
                           (self.namespace, 'exchange_risk_checkpoint', json.dumps({'equity': risk['equity']})))
        if risk['reason'] == 'drawdown_limit':
            connection.execute('INSERT INTO journal(namespace,event,payload) VALUES (?,?,?)',
                               (self.namespace, 'exchange_entry_stop', json.dumps({'reason': 'drawdown_limit'})))


    def _entry_control(self, connection: sqlite3.Connection) -> dict[str, Any]:
        row = connection.execute("SELECT payload FROM journal WHERE namespace=? AND event='entry_control' ORDER BY sequence DESC LIMIT 1", (self.namespace,)).fetchone()
        values = json.loads(row[0]) if row else {'paused': False}
        if type(values.get('paused')) is not bool:
            raise StateUnavailable('Invalid durable entry control')
        return values

    def set_entry_pause(self, paused: bool, *, actor: str) -> None:
        if self.mode != 'exchange' or type(paused) is not bool or not actor:
            raise ConfigurationError('Explicit exchange entry control required')
        with self.connection(write=True) as connection:
            connection.execute('INSERT INTO journal(namespace,event,payload) VALUES (?,?,?)',
                               (self.namespace, 'entry_control', json.dumps({'paused': paused, 'actor': actor})))

    def automation(self) -> dict[str, Any]:
        with self.connection() as connection:
            return self._automation(connection)

    def _automation(self, connection: sqlite3.Connection) -> dict[str, Any]:
        row = connection.execute("SELECT payload FROM journal WHERE namespace=? AND event='automation_status' ORDER BY sequence DESC LIMIT 1", (self.namespace,)).fetchone()
        if row is None:
            return {'state':'disabled','enabled':False,'reason':'not_started'}
        values = json.loads(row[0])
        if not isinstance(values,dict) or values.get('state') not in {'disabled','waiting_for_reconciliation','blocked','running','stopped'} or type(values.get('enabled')) is not bool:
            raise StateUnavailable('Invalid automation status')
        owner = self._ownership(connection)
        if values['enabled'] and values['state'] != 'stopped':
            if owner.get('state') == 'stopped':
                return {**values,'state':'stopped','reason':'owner_stopped'}
            if values.get('owner_generation') != owner.get('generation'):
                return {**values,'state':'waiting_for_reconciliation','reason':'owner_changed'}
        return values

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        """One committed observation for an entire HTTP request, including joins."""
        with self.connection() as connection:
            connection.execute('BEGIN')
            version, config = self._config(connection)
            return {'config_version': version, 'config': config, 'scan': self._scan(connection),
                    'automation':self._automation(connection),'positions': self._positions(connection), 'operations': self._operations(connection),
                    'migration': self._migration(connection), 'ownership': self._ownership(connection),
                    'market_prices': self._prices(connection), 'market_budget': self._market_budget(connection),
                    'fills': self._fills(connection), 'protection': self._protection(connection), 'position_rules': self._position_rules(connection), 'paper_account': self._paper_account(connection),
                    'exchange_account': self._exchange_account(connection), 'entry_control': self._entry_control(connection),
                    'exchange_risk': self.exchange_risk(connection, now=time.time() if now is None else now) if self.mode == 'exchange' else {'state': 'not_applicable'}}

    def _protection(self, connection: sqlite3.Connection) -> list[dict[str, Any]]:
        groups = []
        for row in connection.execute('SELECT * FROM protection_groups WHERE namespace=? ORDER BY rowid', (self.namespace,)):
            group = dict(row)
            quantity = Decimal(group['quantity'])
            if not quantity.is_finite() or quantity <= 0 or group['state'] not in {'installing', 'unknown', 'active', 'failed', 'canceling', 'closed', 'canceled'}:
                raise StateUnavailable('Invalid protection projection')
            group['quantity'] = decimal_text(quantity)
            for event in connection.execute("SELECT payload FROM journal WHERE namespace=? AND operation_id=? AND event='protection_observed' ORDER BY sequence", (self.namespace, group['group_id'])):
                group.update(json.loads(event[0]))
            group['legs'] = [dict(leg) for leg in connection.execute("SELECT operation_id,provider_client_id,provider_order_id,state,requested_qty FROM intents WHERE namespace=? AND request_hash=? ORDER BY rowid", (self.namespace, 'protection:'+group['group_id']))]
            groups.append(group)
        return groups
