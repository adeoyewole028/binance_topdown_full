"""Kernel-held single-host ownership; no time-expiry or process-global fencing."""
from __future__ import annotations

import hashlib
import os
import sqlite3
from importlib import import_module
from types import TracebackType
from typing import BinaryIO, Self
import uuid

from state_store import StateStore, StateUnavailable


class OwnershipUnavailable(StateUnavailable):
    """No new dispatch is permitted without confirmed exclusive ownership."""


class ExecutionOwner:
    def __init__(self, store: StateStore) -> None:
        self.store = store
        self.token = uuid.uuid4().hex
        self.lock_file: BinaryIO | None = None
        self.generation = 0
        identity = hashlib.sha256(f'{store.account}:{store.venue}'.encode()).hexdigest()
        self.lock_path = store.path.parent / f'owner-{identity}.lock'

    def __enter__(self) -> Self:
        try:
            self.store.validate()
            self.lock_file = self.lock_path.open('a+b', buffering=0)
            self.lock_file.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                unix_locking = import_module('fcntl')
                unix_locking.flock(self.lock_file.fileno(), unix_locking.LOCK_EX | unix_locking.LOCK_NB)
            with self.store.connection(write=True) as connection:
                prior = connection.execute('SELECT generation FROM execution_owners WHERE account=? AND venue=?',
                                           (self.store.account, self.store.venue)).fetchone()
                self.generation = (int(prior[0]) if prior is not None else 0) + 1
                # Recovery is an explicit subsequent step. A free OS lock never
                # proves that a previous in-flight provider request had no effect.
                connection.execute('''INSERT INTO execution_owners(account,venue,token,generation,state) VALUES (?,?,?,?,'recovering')
                    ON CONFLICT(account,venue) DO UPDATE SET token=excluded.token,generation=excluded.generation,
                    state='recovering',updated_at=CURRENT_TIMESTAMP''',
                                   (self.store.account, self.store.venue, self.token, self.generation))
            return self
        except (OSError, StateUnavailable) as error:
            if self.lock_file is not None:
                self.lock_file.close()
                self.lock_file = None
            raise OwnershipUnavailable('Execution ownership unavailable') from error

    def ownership_confirmed(self) -> bool:
        if self.lock_file is None or self.lock_file.closed:
            return False
        with self.store.connection() as connection:
            return self._matches_owner(connection)

    def _matches_owner(self, connection: sqlite3.Connection) -> bool:
        if self.lock_file is None or self.lock_file.closed:
            return False
        try:
            handle = os.fstat(self.lock_file.fileno())
            current = self.lock_path.stat()
            if (handle.st_dev, handle.st_ino) != (current.st_dev, current.st_ino):
                return False
        except OSError:
            return False
        row = connection.execute('SELECT token,generation FROM execution_owners WHERE account=? AND venue=?',
                                 (self.store.account, self.store.venue)).fetchone()
        return row is not None and row[0] == self.token and row[1] == self.generation

    def __exit__(self, error_type: type[BaseException] | None,
                 error: BaseException | None, traceback: TracebackType | None) -> None:
        try:
            with self.store.connection(write=True) as connection:
                connection.execute('''UPDATE execution_owners SET state='stopped',updated_at=CURRENT_TIMESTAMP
                                    WHERE account=? AND venue=? AND token=? AND generation=?''',
                                   (self.store.account, self.store.venue, self.token, self.generation))
        finally:
            if self.lock_file is not None:
                self.lock_file.close()
                self.lock_file = None
