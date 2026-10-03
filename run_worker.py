"""Explicit single-host executor launcher; supervisor installation is BT14."""
from __future__ import annotations

import argparse
import os
import time

from execution_worker import ExecutionWorker
from runtime_environment import load_launcher_environment
from state_store import StateStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', action='store_true', help='Run advisory scanner under the same ownership lock')
    parser.add_argument('--once', action='store_true', help='Process one queued batch then stop')
    arguments = parser.parse_args()
    runtime = load_launcher_environment()
    store = StateStore(runtime.data_directory, os.environ.get('LOCAL_ACCOUNT_ID', 'operator-account'), runtime.venue, runtime.mode)
    provider = None
    if runtime.mode == 'exchange':
        from exchange_adapter import BinanceSpotOrders
        provider = BinanceSpotOrders(runtime)
    with ExecutionWorker(store, exchange_provider=provider) as owner:
        if arguments.scan:
            if arguments.once:
                raise ValueError('A scanner is a persistent owner; --once is incompatible')
            from dashboard import initialize_runtime, bot_loop
            initialize_runtime()
            bot_loop(owner)
        else:
            while True:
                owner.process_pending()
                if arguments.once:
                    return
                time.sleep(1)


if __name__ == '__main__':
    main()
