"""BT12 truthful rendering and settings through the authenticated HTTP seam."""
from tests.legacy_sandbox import LegacySandbox
import pytest


def test_dashboard_has_local_assets_and_distinct_unavailable_scan_state() -> None:
    with LegacySandbox() as sandbox:
        html=sandbox.client.get('/').get_data(as_text=True)
        assert 'http-equiv="refresh"' not in html
        assert 'cdn.tailwindcss.com' not in html
        assert '/static/dashboard.css' in html
        assert '/static/dashboard.js' in html
        assert 'Market data unavailable' in html
        payload=sandbox.client.get('/api/status').get_json()
        assert payload['namespace']
        assert payload['presentation']['state']=='unavailable'
        assert sandbox.client.get('/static/dashboard.js').mimetype=='text/javascript'
        assert sandbox.client.get('/static/dashboard.css').mimetype=='text/css'


def test_invalid_html_settings_preserve_values_and_show_associated_errors() -> None:
    with LegacySandbox() as sandbox:
        observed=sandbox.client.get('/api/status').get_json()
        before=observed['config']
        response=sandbox.client.post('/update-config',data={'csrf_token':sandbox.csrf,
            'risk':'-2','rsi_threshold':'47','expected_config_version':observed['config_version']},headers={'Accept':'text/html'})
        assert response.status_code==422
        html=response.get_data(as_text=True)
        assert 'value="-2"' in html
        assert 'value="47"' in html
        assert 'aria-invalid="true"' in html
        assert 'role="alert"' in html
        assert sandbox.client.get('/api/status').get_json()['config']==before


def test_json_settings_acknowledges_committed_version_without_weakening_mode() -> None:
    with LegacySandbox() as sandbox:
        observed=sandbox.client.get('/api/status').get_json()
        response=sandbox.client.post('/update-config',data={'csrf_token':sandbox.csrf,
            'risk':'0.023','expected_config_version':observed['config_version']},headers={'Accept':'application/json'})
        assert response.status_code==200
        version=response.get_json()['saved_config_version']
        current=sandbox.client.get('/api/status').get_json()
        assert current['config_version']==version==observed['config_version']+1
        assert current['config']['risk']==0.023
        assert current['execution_mode']=='paper'
        assert current['venue']==observed['venue']


@pytest.mark.parametrize('paper',[True,False])
def test_setup_validation_identifies_fields_and_preserves_html_draft(paper: bool) -> None:
    from tests.test_paper_execution import PAPER_INPUTS
    with LegacySandbox(paper_setup=False,environment={} if paper else {'PAPER_TRADING':'false','ALLOW_EXCHANGE_EXECUTION':'true'}) as sandbox:
        before=sandbox.client.get('/api/status').get_json()
        values={**PAPER_INPUTS,'fee_fraction':'1','fixed_quote':'123',
                'csrf_token':sandbox.csrf,'expected_config_version':before['config_version']}
        route='/paper/setup' if paper else '/exchange/setup'
        response=sandbox.client.post(route,data=values,headers={'Accept':'application/json'})
        assert response.status_code==422
        assert set(response.get_json()['fields'])=={'fee_fraction'}
        assert 'less than 1' in response.get_json()['fields']['fee_fraction']
        html=sandbox.client.post(route,data=values,headers={'Accept':'text/html'})
        assert html.status_code==422
        assert 'value="123"' in html.get_data(as_text=True)
        assert 'aria-describedby="setup-error-fee_fraction"' in html.get_data(as_text=True)
        assert sandbox.client.get('/api/status').get_json()['config']==before['config']


def test_position_and_account_displays_use_qualified_current_performance() -> None:
    from tests.test_market_runtime import ControlledClock, MarketFixture
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock=ControlledClock()
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
            MarketRuntime(owner,MarketFixture(clock()),clock=clock).scan_once()
            owner.publish_prices({'BTC/USDT':100})
            sandbox.client.post('/trade',data={'csrf_token':sandbox.csrf,'symbol':'BTC/USDT','side':'buy','qty':'1'})
            owner.process_pending()
            owner.publish_prices({'BTC/USDT':101})
            fresh=sandbox.client.get('/api/status').get_json()
            assert fresh['open_positions'][0]['pnl']==1
            assert fresh['open_positions'][0]['currency']=='USDT'
            clock.value+=61
            stale=sandbox.client.get('/api/status').get_json()
            assert stale['performance']['equity'] is None
            assert stale['open_positions'][0]['pnl'] is None
            html=sandbox.client.get('/').get_data(as_text=True)
            assert 'Estimated unrealized' in html
            assert 'Estimated equity: Unavailable USDT' in html
