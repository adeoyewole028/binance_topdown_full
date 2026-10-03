"""BT12 truthful rendering and settings through the authenticated HTTP seam."""
from tests.legacy_sandbox import LegacySandbox


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
