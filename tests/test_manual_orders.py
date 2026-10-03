"""BT13 review and recovery through authenticated HTTP/real owned state."""
from typing import Any
from tests.legacy_sandbox import LegacySandbox
from tests.test_market_runtime import ControlledClock, MarketFixture
from execution_worker import ExecutionWorker
from market_runtime import MarketRuntime
import pytest


def review(sandbox: LegacySandbox, **changes: str) -> Any:
    observed=sandbox.client.get('/api/status').get_json()
    return sandbox.client.post('/orders/review',data={
        'csrf_token':sandbox.csrf,'namespace':observed['namespace'],'mode':observed['execution_mode'],
        'venue':observed['venue'],'kind':'trade','symbol':'BTC/USDT','side':'buy','qty':'1',
        'sizing':'quantity','idempotency_key':'bt13-manual',**changes},headers={'Accept':'application/json'})


def test_readonly_review_uses_server_sizing_then_stable_intent_reports_confirmed_fill() -> None:
    clock=ControlledClock()
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
            MarketRuntime(owner,MarketFixture(clock()),clock=clock).scan_once()
            before=sandbox.client.get('/api/status').get_json()
            response=review(sandbox,sizing='fixed',qty='99999')
            assert response.status_code==200
            receipt=response.get_json()
            assert receipt['preview']['quantity']=='1'
            assert receipt['preview']['estimated_fee']=='0'
            assert receipt['preview']['fee_asset']=='USDT'
            assert receipt['preview']['mode']=='paper'
            assert receipt['preview']['config_version']==before['config_version']
            assert sandbox.client.get('/api/status').get_json()['operations']==before['operations']==[]
            assert sandbox.client.get('/api/status').get_json()['paper_account']==before['paper_account']
            submitted={'csrf_token':sandbox.csrf,'review_token':receipt['review_token']}
            first=sandbox.client.post('/orders/confirm',data=submitted,headers={'Accept':'application/json'})
            assert first.status_code==202
            operation=first.get_json()['operation_id']
            assert first.get_json()['operation']['state']=='queued'
            repeated=sandbox.client.post('/orders/confirm',data=submitted,headers={'Accept':'application/json'})
            assert repeated.get_json()['operation_id']==operation
            queued=sandbox.client.get('/api/operations/'+operation).get_json()['operation']
            assert queued['fills']==[] and queued['state']=='queued'
            owner.process_pending()
            filled=sandbox.client.get('/api/operations/'+operation).get_json()['operation']
            assert filled['state']=='filled' and len(filled['fills'])==1
            assert filled['fills'][0]['qty']=='1'
            assert sandbox.client.get('/api/status').get_json()['paper_account']['balances']['USDT']['total']=='9900'
            assert len(sandbox.client.get('/api/status').get_json()['operations'])==1


@pytest.mark.parametrize('change',['price','config','stale','namespace','quantity'])
def test_changed_review_requires_new_review_without_intake(change: str) -> None:
    clock=ControlledClock()
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
            MarketRuntime(owner,MarketFixture(clock()),clock=clock).scan_once()
            response=review(sandbox)
            assert response.status_code==200
            payload={'csrf_token':sandbox.csrf,'review_token':response.get_json()['review_token']}
            if change=='price':owner.publish_prices({'BTC/USDT':101})
            elif change=='config':sandbox.post('/update-config',data={'risk':'0.005'})
            elif change=='stale':clock.value+=61
            elif change=='namespace':payload['namespace']='another-account'
            else:payload['qty']='2'
            refused=sandbox.client.post('/orders/confirm',data=payload,headers={'Accept':'application/json'})
            assert refused.status_code==409
            observed=sandbox.client.get('/api/status').get_json()
            assert observed['operations']==[] and observed['positions']==[]
            assert observed['paper_account']['balances']['USDT']['total']=='10000'


def test_reviewed_funds_cannot_change_between_preview_and_confirm() -> None:
    clock=ControlledClock()
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
            MarketRuntime(owner,MarketFixture(clock()),clock=clock).scan_once()
            receipt=review(sandbox).get_json()
            sandbox.client.post('/trade',data={'csrf_token':sandbox.csrf,'symbol':'BTC/USDT',
                'side':'buy','qty':'1','idempotency_key':'concurrent-order'})
            response=sandbox.client.post('/orders/confirm',data={'csrf_token':sandbox.csrf,
                'review_token':receipt['review_token']},headers={'Accept':'application/json'})
            assert response.status_code==409
            observed=sandbox.client.get('/api/status').get_json()
            assert len(observed['operations'])==1
            assert observed['paper_account']['balances']['USDT']['reserved']=='100'
            assert observed['positions']==[]


def test_reviewed_partial_close_needs_no_exit_signal_and_preserves_protection() -> None:
    clock=ControlledClock()
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
            MarketRuntime(owner,MarketFixture(clock()),clock=clock).scan_once()
            entry=review(sandbox).get_json()
            sandbox.client.post('/orders/confirm',data={'csrf_token':sandbox.csrf,'review_token':entry['review_token']})
            owner.process_pending()
            assert sandbox.client.get('/api/status').get_json()['exit_signals']==[]
            close=review(sandbox,kind='close',side='sell',qty='0.4',idempotency_key='bt13-close').get_json()
            accepted=sandbox.client.post('/orders/confirm',data={'csrf_token':sandbox.csrf,'review_token':close['review_token']}).get_json()
            assert accepted['operation']['state']=='queued'
            owner.process_pending()
            operation=sandbox.client.get('/api/operations/'+accepted['operation_id']).get_json()['operation']
            assert operation['state']=='filled' and operation['fills'][0]['qty']=='0.4'
            active=[group for group in operation['protection'] if group['state']=='active']
            assert len(active)==1 and active[0]['quantity']=='0.6'
            assert sandbox.client.get('/api/status').get_json()['positions'][0]['qty']=='0.6'


@pytest.mark.parametrize('outcome',['partial','unknown','reject','protection_failed','cancel_unknown'])
def test_private_provider_outcomes_have_distinct_recovery_without_replacement(outcome: str) -> None:
    from tests.test_exchange_execution import SpotFixture, configure
    from tests.test_position_protection import enable_protection
    from exchange_adapter import SubmissionRejected
    class Provider(SpotFixture):
        def submit(self,symbol: str,side: str,quantity: str,client_id: str) -> dict[str,Any]:
            if outcome=='reject':raise SubmissionRejected('invalid_market')
            order=super().submit(symbol,side,'0.4' if outcome=='partial' else quantity,client_id)
            if outcome=='partial':
                order.update(origQty=quantity,status='PARTIALLY_FILLED');self.orders[client_id]=order
            if outcome=='unknown':raise TimeoutError('Offline uncertain response')
            return order
        def query(self,symbol: str,client_id: str) -> dict[str,Any] | None:
            return None if outcome=='unknown' else super().query(symbol,client_id)
        def protect(self,*args: Any) -> dict[str,Any]:
            if outcome=='protection_failed':raise SubmissionRejected('invalid_market')
            return super().protect(*args)
    with LegacySandbox(paper_setup=False,environment={'PAPER_TRADING':'false','ALLOW_EXCHANGE_EXECUTION':'true'}) as sandbox:
        configure(sandbox);enable_protection(sandbox);sandbox.seed_price('BTC/USDT',100)
        provider=Provider()
        with ExecutionWorker(sandbox.module.state_store,exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            response=review(sandbox)
            assert response.status_code==200,response.get_json()
            token=response.get_json()['review_token']
            payload={'csrf_token':sandbox.csrf,'review_token':token}
            response=sandbox.client.post('/orders/confirm',data=payload)
            assert response.status_code==202,response.get_json()
            operation=response.get_json()['operation_id'];owner.process_pending()
            recovery=sandbox.client.get('/api/operations/'+operation).get_json()['operation']
            expected={'partial':'partially_filled','unknown':'unknown','reject':'rejected','protection_failed':'filled','cancel_unknown':'filled'}[outcome]
            assert recovery['state']==expected
            if outcome in {'unknown','reject'}:assert recovery['fills']==[]
            else:assert recovery['fills'][0]['qty']==('0.4' if outcome=='partial' else '1')
            if outcome=='protection_failed':
                assert recovery['protection'][-1]['state']=='failed'
                assert 'Protection needs recovery' in recovery['next_action']
                assert recovery['can_start_another'] is False
            if outcome in {'unknown','partial'}:assert recovery['can_start_another'] is False
            repeated=sandbox.client.post('/orders/confirm',data=payload)
            assert repeated.get_json()['operation_id']==operation
            assert len(sandbox.client.get('/api/status').get_json()['operations'])==1
            if outcome=='cancel_unknown':
                provider.timeout_cancel=True
                close=review(sandbox,kind='close',side='sell',qty='0.4',idempotency_key='bt13-cancel-close')
                assert close.status_code==200,close.get_json()
                close_payload={'csrf_token':sandbox.csrf,'review_token':close.get_json()['review_token']}
                accepted=sandbox.client.post('/orders/confirm',data=close_payload).get_json()
                owner.process_pending()
                pending=sandbox.client.get('/api/operations/'+accepted['operation_id']).get_json()['operation']
                assert pending['fills']==[] and pending['state']=='queued'
                assert pending['protection'][-1]['state']=='canceling'
                assert pending['can_start_another'] is False
                assert len(provider.submissions)==1
                assert sandbox.client.post('/orders/confirm',data=close_payload).get_json()['operation_id']==accepted['operation_id']
                owner.process_pending()
                assert len(provider.submissions)==1


def test_review_permission_and_mode_conflict_preserve_account_and_signed_retry() -> None:
    clock=ControlledClock()
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
            MarketRuntime(owner,MarketFixture(clock()),clock=clock).scan_once()
            receipt=review(sandbox).get_json()
            assert review(sandbox,mode='exchange').status_code==409
            with sandbox.client.session_transaction() as session:session.clear()
            response=sandbox.client.post('/orders/confirm',data={'csrf_token':sandbox.csrf,'review_token':receipt['review_token']},headers={'Accept':'application/json'})
            assert response.status_code==401
            sandbox.login()
            assert sandbox.client.get('/api/status').get_json()['operations']==[]
            assert sandbox.client.get('/api/operations/not-in-this-namespace').status_code==404


@pytest.mark.parametrize('invalid',['signature','reused_token','invalid_quantity','wrong_venue'])
def test_invalid_or_reused_review_keeps_one_authorized_intent(invalid: str) -> None:
    clock=ControlledClock()
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
            MarketRuntime(owner,MarketFixture(clock()),clock=clock).scan_once()
            receipt=review(sandbox).get_json()
            if invalid=='signature':
                response=sandbox.client.post('/orders/confirm',data={'csrf_token':sandbox.csrf,'review_token':receipt['review_token']+'tampered'})
                assert response.status_code==409
            elif invalid=='reused_token':
                original=sandbox.client.post('/orders/confirm',data={'csrf_token':sandbox.csrf,'review_token':receipt['review_token']}).get_json()
                assert sandbox.client.get('/api/order-token/bt13-manual').get_json()['operation']['operation_id']==original['operation_id']
                other=review(sandbox,qty='2').get_json()
                assert sandbox.client.post('/orders/confirm',data={'csrf_token':sandbox.csrf,'review_token':other['review_token']}).status_code==409
            else:
                response=review(sandbox,**({'qty':'NaN'} if invalid=='invalid_quantity' else {'venue':'mainnet'}))
                assert response.status_code==(422 if invalid=='invalid_quantity' else 409)
            observed=sandbox.client.get('/api/status').get_json()
            assert len(observed['operations'])==(1 if invalid=='reused_token' else 0)
            assert observed['positions']==[] and observed['fills']==[]
