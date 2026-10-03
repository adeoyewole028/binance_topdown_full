"""Explicit offline migration operations. Nothing runs during application import."""
from __future__ import annotations

from collections import Counter
from contextlib import closing
from decimal import Decimal, InvalidOperation
import hashlib
import csv
import io
import json
from pathlib import Path
import stat
import sqlite3
from typing import Any

from execution_owner import ExecutionOwner
from state_store import StateStore, StateUnavailable


def backup_state(store: StateStore, destination: Path) -> None:
    """Consistent online SQLite backup, checksum, and independent store identity."""
    store.validate()
    destination.mkdir(parents=True, exist_ok=False)
    target = destination / 'trading.sqlite3'
    with store.connection() as source:
        with closing(sqlite3.connect(target)) as output:
            source.backup(output)
            if output.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise StateUnavailable('Backup integrity failure')
            identity = output.execute('SELECT id FROM store_identity').fetchone()[0]
            sequence = output.execute('SELECT coalesce(max(sequence),0) FROM journal').fetchone()[0]
    marker = {'id': identity, 'schema_version': 1}
    (destination / 'state-identity.json').write_text(json.dumps(marker), encoding='utf-8')
    manifest = {**marker, 'journal_sequence': sequence,
                'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'complete': True}
    (destination / 'backup-manifest.json').write_text(json.dumps(manifest), encoding='utf-8')


def restore_state(current: StateStore, backup: Path, destination: Path) -> None:
    """Prepare an isolated candidate. Never overwrite the authoritative journal.

    An older backup is refused while newer effects exist. Rollback uses a
    compatible reader and retains that database; it cannot resume JSON writers.
    Actual service pause/cutover is an independently authorized operational gate.
    """
    try:
        manifest = json.loads((backup / 'backup-manifest.json').read_text(encoding='utf-8'))
        raw = (backup / 'trading.sqlite3').read_bytes()
        if manifest.get('complete') is not True or hashlib.sha256(raw).hexdigest() != manifest['sha256']:
            raise StateUnavailable('Backup checksum mismatch')
        if current.path.exists() or current.marker.exists():
            with ExecutionOwner(current):
                with current.connection() as source:
                    identity = source.execute('SELECT id FROM store_identity').fetchone()[0]
                    sequence = source.execute('SELECT coalesce(max(sequence),0) FROM journal').fetchone()[0]
                    if identity != manifest['id']:
                        raise StateUnavailable('Backup belongs to another store')
                    if sequence > manifest['journal_sequence']:
                        raise StateUnavailable('Restore would discard newer journal effects; retain authoritative store')
        destination.mkdir(parents=True, exist_ok=False)
        target = destination / 'trading.sqlite3'
        with target.open('xb') as output:
            output.write(raw)
        (destination / 'state-identity.json').write_text(json.dumps({'id': manifest['id'], 'schema_version': 1}), encoding='utf-8')
        restored = StateStore(destination, current.account, current.venue, current.mode)
        restored.validate()
    except (OSError, ValueError, KeyError, sqlite3.Error) as error:
        raise StateUnavailable('Restore candidate unavailable') from error


def quarantine_legacy(store: StateStore, sources: list[Path]) -> list[dict[str, Any]]:
    """Checksum/copy first, then journal quarantine. Unknown origin has no effects."""
    with ExecutionOwner(store):
        for source in sources:
            raw = source.read_bytes()
            checksum = hashlib.sha256(raw).hexdigest()
            if source.name != 'positions.json' and source.name != 'config.json' and source.suffix.lower() != '.csv':
                raise StateUnavailable('Only legacy positions, config and CSV inputs are accepted')
            kind = 'positions' if source.name == 'positions.json' else 'review-only'
            try:
                if source.suffix.lower() == '.csv':
                    count = len(list(csv.DictReader(io.StringIO(raw.decode('utf-8')))))
                else:
                    decoded = json.loads(raw)
                    count = len(decoded) if isinstance(decoded, (list, dict)) else 1
            except (ValueError, UnicodeError):
                count = 1
            backup = store.path.parent / 'legacy-quarantine' / checksum / source.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists():
                if hashlib.sha256(backup.read_bytes()).hexdigest() != checksum:
                    raise StateUnavailable('Quarantine backup checksum mismatch')
            else:
                with backup.open('xb') as output:
                    output.write(raw)
                    output.flush()
                    import os
                    os.fsync(output.fileno())
                backup.chmod(stat.S_IREAD)
            # Config and CSV stay in read-only backup; do not copy arbitrary
            # fields/possible secrets into the runtime database.
            contents = json.dumps(_position_records(raw), allow_nan=False) if kind == 'positions' else '[]'
            with store.connection(write=True) as connection:
                existed = connection.execute('SELECT 1 FROM migration_sources WHERE namespace=? AND checksum=?',
                                             (store.namespace, checksum)).fetchone()
                if existed is None:
                    connection.execute('''INSERT INTO migration_sources(namespace,checksum,source_name,kind,state,
                                          record_count,quarantine_count,contents) VALUES (?,?,?,?,'quarantined',?,?,?)''',
                                       (store.namespace, checksum, source.name, kind, count, count, contents))
                    connection.execute('INSERT INTO journal(namespace,event,payload) VALUES (?, ?, ?)',
                                       (store.namespace, 'legacy_quarantined', json.dumps({'checksum': checksum, 'count': count})))
            source.chmod(stat.S_IREAD)
    return store.migration()


def _position_records(raw: bytes) -> list[dict[str, str]]:
    try:
        records = json.loads(raw)
    except (ValueError, UnicodeError):
        return []
    if not isinstance(records, list):
        return []
    # Only recognized non-secret position fields enter the runtime store.
    return [{key: str(record.get(key, '')) for key in ('symbol', 'qty', 'entry')}
            if isinstance(record, dict) else {'symbol': '', 'qty': '', 'entry': ''}
            for record in records]


def confirm_paper_snapshot(store: StateStore, checksum: str, *, actor: str,
                           origin_confirmation: str) -> None:
    """Operator attestations bind source and namespace; never manufacture fills."""
    if store.mode != 'paper' or origin_confirmation != f'paper:{store.namespace}:{checksum}' or not actor:
        raise StateUnavailable('Exact paper-origin confirmation required')
    with ExecutionOwner(store):
        with store.connection(write=True) as connection:
            row = connection.execute('SELECT * FROM migration_sources WHERE namespace=? AND checksum=?',
                                     (store.namespace, checksum)).fetchone()
            if row is None or row['kind'] != 'positions':
                raise StateUnavailable('A quarantined position snapshot is required')
            if row['state'] == 'reviewed-paper':
                return
            records = json.loads(row['contents'])
            counts = Counter(record['symbol'] for record in records)
            imported = 0
            for record in records:
                symbol = record['symbol']
                try:
                    quantity, entry = Decimal(record['qty']), Decimal(record['entry'])
                    valid = quantity.is_finite() and entry.is_finite() and quantity > 0 and entry > 0
                except InvalidOperation:
                    valid = False
                # All copies of duplicate symbols are anomalies, not addable lots.
                if not valid or counts[symbol] != 1 or symbol.count('/') != 1 or not all(symbol.split('/')):
                    continue
                exists = connection.execute('SELECT 1 FROM positions WHERE namespace=? AND symbol=?',
                                            (store.namespace, symbol)).fetchone()
                if exists is not None:
                    continue  # Overlap requires review; never sum two snapshots.
                connection.execute('INSERT INTO positions VALUES (?,?,?,?,?)',
                                   (store.namespace, symbol, str(quantity), str(entry), 'legacy-opening; fees/capital/history unknown'))
                imported += 1
            connection.execute('''UPDATE migration_sources SET state='reviewed-paper',imported_count=?,quarantine_count=?
                                  WHERE namespace=? AND checksum=?''',
                               (imported, row['record_count'] - imported, store.namespace, checksum))
            connection.execute('INSERT INTO journal(namespace,event,payload) VALUES (?,?,?)',
                               (store.namespace, 'legacy_paper_opening', json.dumps({'checksum': checksum, 'actor': actor,
                                'origin_confirmation': origin_confirmation, 'imported_count': imported,
                                'quarantine_count': row['record_count'] - imported, 'history_complete': False})))
