"""Disposable subprocess used to prove kernel ownership and crash handover."""
from pathlib import Path
import sys

from execution_owner import OwnershipUnavailable
from execution_worker import ExecutionWorker
from state_store import StateStore


def main() -> int:
    store = StateStore(Path(sys.argv[1]), 'operator-account', 'testnet', 'paper')
    try:
        with ExecutionWorker(store) as worker:
            print('owned', flush=True)
            if sys.argv[2] == 'hold':
                sys.stdin.readline()
            worker.process_pending()
    except OwnershipUnavailable:
        print('ownership-unavailable', flush=True)
        return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
