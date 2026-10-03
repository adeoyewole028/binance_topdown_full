"""Disposable automation owner for actual kernel-fencing/termination proof."""
from pathlib import Path
import sys

from automation import Automation
from execution_worker import ExecutionWorker
from state_store import StateStore


def main() -> None:
    store=StateStore(Path(sys.argv[1]),'operator-account','testnet','paper')
    with ExecutionWorker(store) as owner:
        Automation(owner,enabled=True)
        print('owned',flush=True)
        sys.stdin.readline()


if __name__ == '__main__':
    main()
