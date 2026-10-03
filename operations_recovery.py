"""Single-host scheduled verified backups; never overwrite live financial state."""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import uuid

from operations_policy import OperationsPolicy
from state_admin import backup_state
from state_store import StateStore, StateUnavailable


class BackupSchedule:
    """Only the service supervisor calls step, independently of scanner work.

    Incomplete attempts remain quarantined. Retention deletes only this manager's
    four verified files in a same-store directory. Host-loss recovery additionally
    requires an independently authorized off-host copy of complete snapshots.
    """
    def __init__(self, store: StateStore) -> None:
        self.store = store
        self.directory = store.path.parent / 'backups'
        self.policy = OperationsPolicy()
        self.latest: float | None = None
        self.interval = self.policy.recovery_point_seconds - 2 * self.policy.reconciliation_interval

    def _publish_health(self, stamp: float, identity: str, snapshot: Path) -> None:
        path = self.store.path.parent / 'backup-health.json'
        temporary = path.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8') as handle:
            details = (snapshot / 'trading.sqlite3').stat()
            handle.write(json.dumps({'as_of': stamp, 'store_id': identity,
                'snapshot': snapshot.name, 'size': details.st_size, 'modified_ns': details.st_mtime_ns}))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    def _verified(self, candidate: Path) -> tuple[float, str]:
        if (candidate.is_symlink() or candidate.resolve().parent != self.directory.resolve()
                or {p.name for p in candidate.iterdir()} != {
                    'trading.sqlite3', 'state-identity.json', 'backup-manifest.json', 'retention.json'}
                or any(p.is_symlink() for p in candidate.iterdir())):
            raise StateUnavailable('Backup layout unavailable')
        metadata = json.loads((candidate / 'retention.json').read_text(encoding='utf-8'))
        manifest = json.loads((candidate / 'backup-manifest.json').read_text(encoding='utf-8'))
        stamp = metadata['created_at']
        if isinstance(stamp, bool) or not isinstance(stamp, (float, int)) or not math.isfinite(stamp) or stamp < 0:
            raise StateUnavailable('Backup time unavailable')
        raw = (candidate / 'trading.sqlite3').read_bytes()
        if manifest.get('complete') is not True or hashlib.sha256(raw).hexdigest() != manifest['sha256']:
            raise StateUnavailable('Backup integrity unavailable')
        with closing(sqlite3.connect((candidate / 'trading.sqlite3').as_uri() + '?mode=ro', uri=True)) as connection:
            identity = connection.execute('SELECT id,schema_version FROM store_identity').fetchone()
            sequence = connection.execute('SELECT coalesce(max(sequence),0) FROM journal').fetchone()[0]
            if (connection.execute('PRAGMA integrity_check').fetchone()[0] != 'ok'
                    or connection.execute('PRAGMA foreign_key_check').fetchone()
                    or identity is None or identity != (manifest['id'], 1) or manifest['schema_version'] != 1
                    or sequence != manifest['journal_sequence'] or metadata['store_id'] != identity[0]):
                raise StateUnavailable('Backup metadata unavailable')
        with self.store.connection() as source:
            if source.execute('SELECT id FROM store_identity').fetchone()[0] != identity[0]:
                raise StateUnavailable('Backup belongs to another store')
        return float(stamp), str(identity[0])

    def step(self, now: float) -> Path | None:
        """Create at startup/900s, verify before claiming success, retain seven days."""
        try:
            if not math.isfinite(now) or now < 0 or self.directory.is_symlink():
                raise StateUnavailable('Backup clock or directory unavailable')
            if self.latest is not None:
                if now < self.latest:
                    raise StateUnavailable('Backup clock moved backwards')
                if now - self.latest < self.interval:
                    return None
            self.store.validate()
            self.directory.mkdir(exist_ok=True)
            snapshots = [(p, self._verified(p)[0]) for p in self.directory.glob('snapshot-*')]
            latest = max((stamp for _, stamp in snapshots), default=None)
            if latest is not None and now < latest:
                raise StateUnavailable('Backup clock moved backwards')
            if latest is not None and now - latest < self.interval:
                self.latest = latest
                newest = max(snapshots, key=lambda pair: pair[1])[0]
                self._publish_health(latest, self._verified(newest)[1], newest)
                return None
            identity = uuid.uuid4().hex
            staging = self.directory / ('.incomplete-' + identity)
            backup_state(self.store, staging)
            manifest = json.loads((staging / 'backup-manifest.json').read_text(encoding='utf-8'))
            (staging / 'retention.json').write_text(json.dumps({
                'created_at': now, 'store_id': manifest['id'],
            }), encoding='utf-8')
            self._verified(staging)
            for file in staging.iterdir():
                with file.open('rb+') as handle:
                    os.fsync(handle.fileno())
            published = self.directory / ('snapshot-' + identity)
            staging.rename(published)
            if os.name != 'nt':
                descriptor = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            cutoff = now - self.policy.backup_retention_days * 86400
            for candidate, stamp in snapshots:
                if stamp < cutoff:
                    self._verified(candidate)
                    for name in ('trading.sqlite3', 'state-identity.json', 'backup-manifest.json', 'retention.json'):
                        (candidate / name).unlink()
                    candidate.rmdir()
            self.latest = now
            self._publish_health(now, manifest['id'], published)
            return published
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
            raise StateUnavailable('Scheduled backup unavailable; execution must stop') from error
