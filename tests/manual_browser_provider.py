"""Controllable private provider fixture; no application collaborators replaced."""
from typing import Any
from tests.test_exchange_execution import SpotFixture
from exchange_adapter import SubmissionRejected


class ManualBrowserProvider(SpotFixture):
    outcome='fill'
    visible=True

    def submit(self,symbol: str,side: str,quantity: str,client_id: str) -> dict[str,Any]:
        if self.outcome=='reject':raise SubmissionRejected('invalid_market')
        order=super().submit(symbol,side,'0.4' if self.outcome=='partial' else quantity,client_id)
        if self.outcome=='partial':
            order.update(origQty=quantity,status='PARTIALLY_FILLED')
            self.orders[client_id]=order
        if self.outcome=='unknown':
            self.visible=False
            raise TimeoutError('Offline response lost')
        return order

    def query(self,symbol: str,client_id: str) -> dict[str,Any] | None:
        return super().query(symbol,client_id) if self.visible else None

    def protect(self,*args: Any) -> dict[str,Any]:
        if self.outcome=='protection_failed':raise SubmissionRejected('invalid_market')
        return super().protect(*args)
