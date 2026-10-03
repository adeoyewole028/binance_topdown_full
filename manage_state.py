"""Explicit maintenance CLI. Requires stopped writers and reviewed source paths."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from runtime_environment import load_launcher_environment
from state_admin import backup_state, confirm_paper_snapshot, quarantine_legacy, restore_state
from state_store import StateStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('init', help='Explicitly create a new namespace; never repair lost state')
    commands.add_parser('upgrade-market', help='Explicitly install the additive BT04 request-budget extension')
    resume = commands.add_parser('resume-market', help='After reviewing access/auth failure, clear its terminal block; requires stopped owner')
    resume.add_argument('--actor', required=True)
    quarantine = commands.add_parser('quarantine')
    quarantine.add_argument('sources', type=Path, nargs='+')
    confirm = commands.add_parser('confirm-paper')
    confirm.add_argument('checksum')
    confirm.add_argument('--confirmation', required=True)
    confirm.add_argument('--actor', required=True)
    backup = commands.add_parser('backup')
    backup.add_argument('destination', type=Path)
    restore = commands.add_parser('restore-candidate')
    restore.add_argument('backup', type=Path)
    restore.add_argument('destination', type=Path)
    commands.add_parser('inspect')
    arguments = parser.parse_args()
    runtime = load_launcher_environment()
    store = StateStore(runtime.data_directory, os.environ.get('LOCAL_ACCOUNT_ID', 'operator-account'), runtime.venue, runtime.mode)
    if arguments.command == 'init':
        from dashboard import environment_defaults
        store.provision(environment_defaults(os.environ))
    elif arguments.command == 'upgrade-market':
        from execution_owner import ExecutionOwner
        with ExecutionOwner(store):
            store.provision_market()
    elif arguments.command == 'resume-market':
        from execution_owner import ExecutionOwner
        from request_budget import RequestBudget
        with ExecutionOwner(store) as owner:
            RequestBudget(store).resume(owner, actor=arguments.actor)
    elif arguments.command == 'quarantine':
        quarantine_legacy(store, [path.resolve() for path in arguments.sources])
    elif arguments.command == 'confirm-paper':
        confirm_paper_snapshot(store, arguments.checksum, actor=arguments.actor, origin_confirmation=arguments.confirmation)
    elif arguments.command == 'backup':
        backup_state(store, arguments.destination.resolve())
    elif arguments.command == 'restore-candidate':
        restore_state(store, arguments.backup.resolve(), arguments.destination.resolve())
    store.validate()
    print(json.dumps({'namespace': store.namespace, 'mode': store.mode, 'venue': store.venue,
                      'migration': store.migration(), 'ownership': store.ownership()}, indent=2))


if __name__ == '__main__':
    main()
