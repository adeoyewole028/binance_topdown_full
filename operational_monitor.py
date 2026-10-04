"""Durable redacted incident transitions and explicitly authorized delivery."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib import request
from urllib.parse import urlsplit
import uuid

from operational_health import observations
from operations_policy import OperationsPolicy
from operations_recovery import BackupSchedule, sync_directory
from tools.http_transport import NoRedirect
from runtime_environment import parse_boolean
from state_store import StateStore, StateUnavailable


class Monitor:
    def __init__(self, store: StateStore) -> None:
        self.store = store
        self.path = store.path.parent / 'operations-alert.json'
        self.outbox = store.path.parent / 'operations-alerts'
        self.delivered = self.outbox / 'delivered'
        self.next_check = 0.0
        self.backups = BackupSchedule(store)
        self.destination = os.environ.get('OPERATOR_ALERT_WEBHOOK', '')
        self.destination_revision = hashlib.sha256(self.destination.encode()).hexdigest()
        self.authorized = parse_boolean(os.environ.get('OPERATOR_ALERT_DELIVERY_AUTHORIZED', 'false'))
        if self.authorized:
            parsed = urlsplit(self.destination)
            if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
                raise ValueError('An explicitly authorized HTTPS private webhook is required')

    def step(self, *, failure: bool = False) -> None:
        now = time.time()
        elapsed = time.monotonic()
        if elapsed < self.next_check and not failure:
            return
        self.next_check = elapsed + OperationsPolicy().reconciliation_interval
        self.backups.step(now)
        report = observations(self.store, now)
        if failure:
            report['conditions'].append('service_child_failed')
        # No balances, credentials, URL, raw exception/provider messages or
        # arbitrary journal fields enter logs or outbound payloads.
        print(json.dumps({'event': 'operational_health', **report}), flush=True)
        incident = {key: report[key] for key in ('conditions', 'operation_ids', 'protection_ids', 'unprotected_symbols')}
        fingerprint = hashlib.sha256(json.dumps(incident, sort_keys=True).encode()).hexdigest()
        try:
            if self.outbox.is_symlink() or self.delivered.is_symlink():
                raise StateUnavailable('Incident outbox unavailable')
            self.outbox.mkdir(exist_ok=True)
            self.delivered.mkdir(exist_ok=True)
            sync_directory(self.outbox)
            sync_directory(self.outbox.parent)
            saved = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
            if saved.get('fingerprint') != fingerprint:
                saved = {'fingerprint': fingerprint, 'delivered': False,
                         'event_id': uuid.uuid4().hex, 'as_of': now,
                         'destination_revision': self.destination_revision, **incident}
                self._save(saved, self.outbox / (saved['event_id'] + '.json'))
                self._save(saved)
            current_name = saved['event_id'] + '.json'
            current_pending = self.outbox / current_name
            current_delivered = self.delivered / current_name
            if (not re.fullmatch(r'[a-f0-9]{32}', saved['event_id'])
                    or not (current_pending.is_file() or current_delivered.is_file())):
                raise StateUnavailable('Current incident record unavailable')
            queued = []
            for candidate in self.outbox.glob('*.json'):
                if candidate.is_symlink():
                    raise StateUnavailable('Incident outbox unavailable')
                record = json.loads(candidate.read_text(encoding='utf-8'))
                if not re.fullmatch(r'[a-f0-9]{32}', record['event_id']) or candidate.stem != record['event_id']:
                    raise StateUnavailable('Incident identity unavailable')
                if (record.get('delivered') is not True or
                        (record['event_id'] == saved['event_id'] and record.get('destination_revision') != self.destination_revision)):
                    queued.append((candidate, record))
                else:
                    self._archive(candidate)
            if (current_delivered.is_file() and saved.get('delivered') is True
                    and saved.get('destination_revision') != self.destination_revision):
                queued.append((current_delivered, saved))
            # Current actionable transition first; backlog remains durable. One
            # bounded attempt per tick cannot exhaust the supervisor's cadence.
            queued.sort(key=lambda item: (item[1]['event_id'] != saved['event_id'], item[1]['as_of']))
            if self.authorized and queued:
                candidate, record = queued[0]
                try:
                    payload = json.dumps({key: record[key] for key in
                        ('event_id', 'as_of', 'conditions', 'operation_ids', 'protection_ids', 'unprotected_symbols')}).encode()
                    req = request.Request(self.destination, data=payload,
                                          headers={'Content-Type': 'application/json'}, method='POST')
                    with request.build_opener(NoRedirect()).open(req, timeout=5) as response:
                        if not 200 <= response.status < 300:
                            raise StateUnavailable('Alert delivery unavailable')
                    record.update(delivered=True, destination_revision=self.destination_revision)
                    self._save(record, candidate)
                    if record['event_id'] == saved['event_id']:
                        self._save(record)
                    self._archive(candidate)
                except Exception:
                    print(json.dumps({'event': 'alert_delivery_unavailable', 'event_id': record['event_id']}), flush=True)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise StateUnavailable('Operational incident persistence unavailable') from error

    def _archive(self, candidate: Path) -> None:
        candidate.replace(self.delivered / candidate.name)
        sync_directory(self.delivered)
        sync_directory(self.outbox)

    def _save(self, value: dict[str, Any], path: Path | None = None) -> None:
        destination = self.path if path is None else path
        temporary = destination.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8') as output:
            output.write(json.dumps(value))
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(destination)
        sync_directory(destination.parent)
