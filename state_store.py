"""Transactional, namespace-qualified state. Opening never creates missing state."""
from __future__ import annotations

from contextlib import closing, contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation


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
        except (OSError, sqlite3.Error, ValueError, KeyError) as error:
            raise StateUnavailable('Required durable state unavailable') from error

    def provision(self, defaults: dict[str, Any]) -> None:
        """Explicit initialization only. Never overwrite or heal existing state."""
        if self.path.exists() or self.marker.exists():
            with self.connection(write=True) as connection:
                existing = connection.execute('SELECT id FROM namespaces WHERE id=?', (self.namespace,)).fetchone()
                if existing is None:
                    connection.execute('INSERT INTO namespaces VALUES (?,?,?,?)', (self.namespace, self.account, self.venue, self.mode))
                    values = {**defaults, 'paper_trading': self.mode == 'paper'}
                    connection.execute('INSERT INTO config_versions(namespace,version,values_json,actor) VALUES (?,1,?,?)',
                                       (self.namespace, json.dumps(values, sort_keys=True, allow_nan=False), 'explicit-provision'))
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
            values = {**defaults, 'paper_trading': self.mode == 'paper'}
            connection.execute('INSERT INTO config_versions(namespace,version,values_json,actor) VALUES (?,1,?,?)',
                               (self.namespace, json.dumps(values, sort_keys=True, allow_nan=False), 'explicit-provision'))
            connection.commit()
        # If the marker write fails, the database remains non-tradable until
        # reviewed repair; startup never silently provisions over that partial state.
        self.marker.write_text(json.dumps({'id': identity, 'schema_version': 1}), encoding='utf-8')

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
        return int(row['version']), values

    def update_config(self, changes: dict[str, Any], actor: str) -> tuple[int, dict[str, Any]]:
        with self.connection(write=True) as connection:
            row = connection.execute('SELECT version,values_json FROM config_versions WHERE namespace=? ORDER BY version DESC LIMIT 1',
                                     (self.namespace,)).fetchone()
            if row is None:
                raise StateUnavailable('Namespace requires explicit initialization')
            values = json.loads(row['values_json'])
            values.update(changes)
            values['paper_trading'] = self.mode == 'paper'
            version = int(row['version']) + 1
            connection.execute('INSERT INTO config_versions(namespace,version,values_json,actor) VALUES (?,?,?,?)',
                               (self.namespace, version, json.dumps(values, sort_keys=True, allow_nan=False), actor))
            connection.execute('INSERT INTO journal(namespace,event,payload) VALUES (?,?,?)',
                               (self.namespace, 'config_updated', json.dumps({'version': version, 'actor': actor})))
            return version, values

    def validate(self) -> None:
        with self.connection() as connection:
            if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok' or connection.execute('PRAGMA foreign_key_check').fetchone():
                raise StateUnavailable('Durable state integrity failure')
        self.config()
        self.positions()

    def enqueue(self, kind: str, symbol: str, side: str, quantity: str) -> str:
        import hashlib
        operation = uuid.uuid4().hex
        request = json.dumps({'kind': kind, 'symbol': symbol, 'side': side, 'qty': quantity}, sort_keys=True, allow_nan=False)
        with self.connection(write=True) as connection:
            row = connection.execute('SELECT max(version) FROM config_versions WHERE namespace=?',
                                     (self.namespace,)).fetchone()
            if row is None or row[0] is None:
                raise StateUnavailable('Namespace requires explicit initialization')
            connection.execute('''INSERT INTO intents(namespace,operation_id,idempotency_key,request_hash,kind,symbol,side,
                                   requested_qty,config_version,provider_client_id) VALUES (?,?,?,?,?,?,?,?,?,?)''',
                               (self.namespace, operation, operation, hashlib.sha256(request.encode()).hexdigest(),
                                kind, symbol, side, quantity, row[0], operation))
            event = connection.execute('INSERT INTO journal(namespace,operation_id,event,payload) VALUES (?,?,?,?)',
                                       (self.namespace, operation, 'intent_queued', request))
            connection.execute('UPDATE intents SET intake_sequence=? WHERE namespace=? AND operation_id=?',
                               (event.lastrowid, self.namespace, operation))
        return operation

    def operations(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            return self._operations(connection)

    def _operations(self, connection: sqlite3.Connection) -> list[dict[str, Any]]:
        return [dict(row) for row in connection.execute('''SELECT operation_id,kind,symbol,side,requested_qty,
                       config_version,state,error_class FROM intents WHERE namespace=? ORDER BY intake_sequence''',
                                                       (self.namespace,))]

    def set_price(self, symbol: str, price: str) -> None:
        with self.connection(write=True) as connection:
            connection.execute('''INSERT INTO market_prices VALUES (?,?,?,?)
                                  ON CONFLICT(namespace,symbol) DO UPDATE SET price=excluded.price,fetched_at=excluded.fetched_at''',
                               (self.namespace, symbol, price, datetime.now(timezone.utc).isoformat()))

    def positions(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            return self._positions(connection)

    def _positions(self, connection: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = [dict(row) for row in connection.execute('SELECT symbol,qty,entry,basis_status FROM positions WHERE namespace=? ORDER BY symbol',
                                                      (self.namespace,))]
        try:
            for row in rows:
                for field in ('qty', 'entry'):
                    value = Decimal(row[field])
                    if not value.is_finite() or value <= 0:
                        raise StateUnavailable('Invalid durable position')
        except InvalidOperation as error:
            raise StateUnavailable('Invalid durable position') from error
        return rows

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

    def snapshot(self) -> dict[str, Any]:
        """One committed observation for an entire HTTP request, including joins."""
        with self.connection() as connection:
            connection.execute('BEGIN')
            version, config = self._config(connection)
            return {'config_version': version, 'config': config, 'scan': self._scan(connection),
                    'positions': self._positions(connection), 'operations': self._operations(connection),
                    'migration': self._migration(connection), 'ownership': self._ownership(connection)}
