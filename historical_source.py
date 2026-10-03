"""Public-only history using the same host/venue research request lane."""
from __future__ import annotations
from collections.abc import Callable
from typing import TypeVar
from market_adapter import BinancePublicMarket
from request_budget import RequestBudget, ProviderFailure

T=TypeVar('T')


class BudgetedHistory:
    def __init__(self, provider: BinancePublicMarket, budget: RequestBudget) -> None:
        self.provider,self.budget=provider,budget

    def _request(self, weight: int, call: Callable[[],T], *, discovery: bool=False,
                 raw_requests: int=1) -> T:
        attempted=False
        def attempt() -> T:
            nonlocal attempted
            attempted=True
            return call()
        try:
            return self.budget.request('research',weight,attempt,discovery=discovery,raw_requests=raw_requests)
        finally:
            if attempted:self.budget.observe(self.provider.headers)

    def capture_time(self, symbol: str) -> int:
        started=self.budget.clock()
        metadata=self._request(21,self.provider.metadata,discovery=True,raw_requests=2)
        self.budget.configure(metadata,request_started=started)
        if symbol not in self.provider.symbols:
            raise ProviderFailure('invalid_market')
        stamp=metadata['serverTime']
        if type(stamp) is not int or stamp<0:raise ProviderFailure('data_quality')
        return stamp

    def candles_range(self, symbol: str, timeframe: str, start: int, end: int,
                      limit: int) -> list[list[float]]:
        return self._request(2,lambda:self.provider.candles_range(symbol,timeframe,start,end,limit))
