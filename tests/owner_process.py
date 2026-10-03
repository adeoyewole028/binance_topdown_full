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
            from tests.legacy_sandbox import FakeExchange
            from market_runtime import MarketRuntime
            fixture = FakeExchange()
            fixture.prices = {symbol: float(quote['price']) for symbol, quote in store.prices().items()}
            if fixture.prices:
                MarketRuntime(worker, fixture).scan_once()
            worker.process_pending()
    except OwnershipUnavailable:
        print('ownership-unavailable', flush=True)
        return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
