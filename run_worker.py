"""Explicit single-host executor/automation launcher; supervision is BT14."""
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import os
import signal
import time
from typing import Any

from automation import Automation
from execution_worker import ExecutionWorker
from market_runtime import MarketProvider, MarketRuntime
from runtime_environment import load_launcher_environment, construct_exchange, parse_boolean
from state_store import StateStore
from strategy_profiles import evaluate_observation


def run_owned_loop(owner: ExecutionWorker, provider: MarketProvider | None = None, *,
                   automate: bool = False, sizing: str = 'risk', once: bool = False,
                   should_run: Callable[[], bool], whitelist: set[str] | None = None,
                   blacklist: set[str] | None = None,
                   maintenance: Callable[[], None] | None = None) -> None:
    """Run under an already acquired owner; stop intake at supervisor shutdown."""
    if once and (provider is not None or automate):
        raise ValueError('Scanning/automation requires a persistent owner; --once is incompatible')
    if automate and provider is None:
        raise ValueError('Automation requires owned public scanning')
    control = Automation(owner,enabled=automate,sizing=sizing,intake_allowed=should_run)
    scanner = None if provider is None else MarketRuntime(owner,provider,clock=owner.clock,
        evaluate=evaluate_observation,reconcile=owner.poll_exchange if owner.exchange_execution is not None else None,
        whitelist=whitelist,blacklist=blacklist,automation=control)
    try:
        while should_run():
            if maintenance is not None:
                maintenance()
            if scanner is None:
                owner.process_pending()
            else:
                scanner.step()
            if once:
                return
            time.sleep(1)
    finally:
        control.stop()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan',action='store_true',help='Run advisory scanner under the ownership lock')
    parser.add_argument('--automate',action='store_true',help='Explicit automation; also requires AUTOMATION_ENABLED=true')
    parser.add_argument('--sizing',choices=('fixed','risk'),default='risk',help='Use configured currency allocation or stop-based risk sizing')
    parser.add_argument('--once',action='store_true',help='Process one queued batch then stop')
    parser.add_argument('--operations',action='store_true',help='Supervised backup and heartbeat maintenance')
    arguments = parser.parse_args(argv)
    if arguments.once and (arguments.scan or arguments.automate):
        raise ValueError('Scanning/automation requires a persistent owner; --once is incompatible')
    runtime = load_launcher_environment()
    if arguments.automate and not parse_boolean(os.environ.get('AUTOMATION_ENABLED','false')):
        raise ValueError('Automation is disabled; explicit AUTOMATION_ENABLED=true is required')
    store = StateStore(runtime.data_directory,os.environ.get('LOCAL_ACCOUNT_ID','operator-account'),runtime.venue,runtime.mode)
    running = True
    def stop(signum: int, frame: Any) -> None:
        nonlocal running
        running = False
    handlers = {sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        for sig in handlers:
            signal.signal(sig,stop)
        with ExecutionWorker(store) as owner:
            maintenance = None
            if arguments.operations:
                from operational_health import OwnerMaintenance
                maintenance = OwnerMaintenance(owner).step
            if runtime.mode == 'exchange':
                from exchange_adapter import BinanceSpotOrders
                from exchange_execution import ExchangeExecution
                owner.exchange_execution = ExchangeExecution(owner,BinanceSpotOrders(runtime))
            public = None
            if arguments.scan or arguments.automate:
                from market_adapter import BinancePublicMarket
                public = BinancePublicMarket(construct_exchange(runtime,private=False))
            run_owned_loop(owner,public,automate=arguments.automate,sizing=arguments.sizing,once=arguments.once,
                should_run=lambda: running,
                whitelist={s.strip() for s in os.environ.get('WHITELIST','').split(',') if s.strip()},
                blacklist={s.strip() for s in os.environ.get('BLACKLIST','').split(',') if s.strip()},
                maintenance=maintenance)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig,handler)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('Execution owner unavailable; state retained for reconciliation')
        raise SystemExit(1) from None
