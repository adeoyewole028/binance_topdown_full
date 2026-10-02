"""Operator security acceptance at the approved HTTP boundary."""
from tests.legacy_sandbox import LegacySandbox
import re
from operator_security import configure_operator
from werkzeug.security import generate_password_hash
import pytest


def sign_in(sandbox: LegacySandbox) -> str:
    configure_operator(sandbox.module.app, 'operator', generate_password_hash('offline-password'),
                       'offline-session-secret-with-32-characters', sandbox.directory, development=True)
    page = sandbox.client.get('/login')
    token = re.search(r'name="csrf_token" value="([^"]+)"', page.get_data(as_text=True))
    assert token is not None
    response = sandbox.client.post('/login', data={
        'username': 'operator', 'password': 'offline-password', 'csrf_token': token[1],
    })
    assert response.status_code == 302
    page = sandbox.client.get('/')
    token = re.search(r'name="csrf_token" value="([^"]+)"', page.get_data(as_text=True))
    assert token is not None
    return token[1]


def test_anonymous_account_and_control_access_is_denied() -> None:
    with LegacySandbox(authenticated=False) as sandbox:
        assert sandbox.client.get('/api/status').status_code == 401
        assert sandbox.client.get('/').status_code == 302
        for route in ('/trade', '/close', '/update-config', '/reset-config'):
            assert sandbox.client.post(route, data={'risk': '0.5'}).status_code == 401
        assert sandbox.exchange.holdings == {}


def test_signed_in_operator_can_trade_and_sign_out() -> None:
    with LegacySandbox(authenticated=False) as sandbox:
        configure_operator(sandbox.module.app, 'operator', generate_password_hash('offline-password'),
                           'offline-session-secret-with-32-characters', sandbox.directory, development=True)
        page = sandbox.client.get('/login')
        token = re.search(r'name="csrf_token" value="([^"]+)"', page.get_data(as_text=True))
        assert token is not None
        assert sandbox.client.post('/login', data={
            'username': 'operator', 'password': 'offline-password', 'csrf_token': token[1],
        }).status_code == 302
        page = sandbox.client.get('/')
        token = re.search(r'name="csrf_token" value="([^"]+)"', page.get_data(as_text=True))
        assert token is not None
        sandbox.seed_price('BTC/USDT', 100.0)
        assert sandbox.client.post('/trade', data={
            'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': token[1],
        }).status_code == 302
        assert sandbox.client.get('/api/status').get_json()['open_positions'][0]['qty'] == 1.0
        assert sandbox.client.post('/logout', data={'csrf_token': token[1]}).status_code == 302
        assert sandbox.client.get('/api/status').status_code == 401


@pytest.mark.parametrize('route', ['/trade', '/close', '/update-config', '/reset-config'])
def test_invalid_or_cross_site_requests_have_no_state_effects(route: str) -> None:
    with LegacySandbox(authenticated=False) as sandbox:
        csrf = sign_in(sandbox)
        before = sandbox.client.get('/api/status').get_json()
        sandbox.seed_price('BTC/USDT', 100.0)
        data = {'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'risk': '0.2'}
        for token in ('', 'forged'):
            assert sandbox.client.post(route, data={**data, 'csrf_token': token}).status_code == 403
        assert sandbox.client.post(route, data={**data, 'csrf_token': csrf}, headers={'Origin': 'https://other.example'}).status_code == 403
        after = sandbox.client.get('/api/status').get_json()
        assert after['config'] == before['config']
        assert after['open_positions'] == before['open_positions']
        assert sandbox.exchange.holdings == {}


def test_login_is_throttled_across_fresh_clients_and_restart() -> None:
    with LegacySandbox(authenticated=False) as sandbox:
        configure_operator(sandbox.module.app, 'operator', generate_password_hash('offline-password'),
                           'offline-session-secret-with-32-characters', sandbox.directory, development=True)
        for attempt in range(6):
            client = sandbox.module.app.test_client()
            page = client.get('/login')
            token = re.search(r'name="csrf_token" value="([^"]+)"', page.get_data(as_text=True))
            assert token is not None
            response = client.post('/login', data={'username': 'unknown', 'password': 'wrong', 'csrf_token': token[1]})
            assert response.status_code == (401 if attempt < 5 else 429)
        restarted = sandbox.load_module('dashboard')
        configure_operator(restarted.app, 'operator', generate_password_hash('offline-password'),
                           'offline-session-secret-with-32-characters', sandbox.directory, development=True)
        client = restarted.app.test_client()
        token = re.search(r'name="csrf_token" value="([^"]+)"', client.get('/login').get_data(as_text=True))
        assert token is not None
        assert client.post('/login', data={'username': 'operator', 'password': 'offline-password', 'csrf_token': token[1]}).status_code == 429


def test_runtime_initialization_rejects_missing_production_secret() -> None:
    with LegacySandbox(authenticated=False) as sandbox:
        with pytest.raises(ValueError, match='secret'):
            sandbox.module.initialize_runtime({'APP_ENV': 'production', 'OPERATOR_USERNAME': 'operator',
                                              'OPERATOR_PASSWORD_HASH': generate_password_hash('offline-password')})


def test_partial_config_and_reset_preserve_safety_fields() -> None:
    with LegacySandbox(authenticated=False) as sandbox:
        csrf = sign_in(sandbox)
        sandbox.client.post('/update-config', data={'risk': '0.02', 'csrf_token': csrf})
        config = sandbox.client.get('/api/status').get_json()['config']
        assert config['paper_trading'] is True
        assert config['use_dynamic_universe'] is True
        assert config['risk'] == 0.02
        assert sandbox.client.post('/update-config', data={'paper_trading': 'false', 'csrf_token': csrf}).status_code == 409
        sandbox.client.post('/reset-config', data={'csrf_token': csrf})
        assert sandbox.client.get('/api/status').get_json()['config']['paper_trading'] is True
        assert sandbox.exchange.holdings == {}


def test_closed_server_gate_blocks_persisted_or_in_memory_exchange_mode() -> None:
    with LegacySandbox(authenticated=False) as sandbox:
        csrf = sign_in(sandbox)
        sandbox.module.status['config']['paper_trading'] = False
        for route in ('/trade', '/close'):
            assert sandbox.client.post(route, data={
                'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'csrf_token': csrf,
                'ALLOW_EXCHANGE_EXECUTION': 'true',
            }).status_code == 403
        sandbox.client.post('/reset-config', data={'csrf_token': csrf})
        assert sandbox.client.get('/api/status').get_json()['config']['paper_trading'] is False
        assert sandbox.exchange.holdings == {}


def test_mode_transition_cannot_reinterpret_unresolved_exposure() -> None:
    with LegacySandbox() as sandbox:
        sandbox.seed_price('BTC/USDT', 100.0)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'})
        before = sandbox.client.get('/api/status').get_json()
        response = sandbox.post('/change-mode', data={'mode': 'exchange', 'expected_mode': 'paper', 'confirmation': 'exchange'})
        assert response.status_code == 409
        after = sandbox.client.get('/api/status').get_json()
        assert after['config'] == before['config']
        assert after['open_positions'] == before['open_positions']


def test_security_cookie_expiry_and_password_rotation() -> None:
    with LegacySandbox(authenticated=False) as sandbox:
        configure_operator(sandbox.module.app, 'operator', generate_password_hash('offline-password'),
                           'offline-session-secret-with-32-characters', sandbox.directory)
        client = sandbox.module.app.test_client()
        page = client.get('/login')
        cookie = page.headers.get('Set-Cookie', '')
        assert 'Secure' in cookie and 'HttpOnly' in cookie and 'SameSite=Strict' in cookie
        assert 'no-store' in page.headers['Cache-Control']
        csrf = sign_in(sandbox)
        assert sandbox.client.get('/api/status').status_code == 200
        with sandbox.client.session_transaction() as state:
            state['signed_in_at'] -= 1801
        assert sandbox.client.get('/api/status').status_code == 401
        sign_in(sandbox)
        configure_operator(sandbox.module.app, 'operator', generate_password_hash('changed-password'),
                           'offline-session-secret-with-32-characters', sandbox.directory)
        assert sandbox.client.get('/api/status').status_code == 401
        assert sandbox.client.post('/trade', data={'csrf_token': csrf}).status_code == 401


def test_login_requires_csrf_and_status_is_same_origin_and_redacted() -> None:
    with LegacySandbox(authenticated=False) as sandbox:
        assert sandbox.client.post('/login', data={'username': 'operator', 'password': 'offline-password'}).status_code == 403
        sign_in(sandbox)
        response = sandbox.client.get('/api/status', headers={'Origin': 'https://other.example'})
        assert 'Access-Control-Allow-Origin' not in response.headers
        for forbidden in ('offline-password', 'offline-key', 'offline-secret', 'OPERATOR_PASSWORD_HASH', 'FLASK_SECRET_KEY'):
            assert forbidden not in response.get_data(as_text=True)


def test_signed_out_session_cannot_be_replayed() -> None:
    with LegacySandbox() as sandbox:
        cookie = sandbox.client.get_cookie('session')
        assert cookie is not None
        stolen = sandbox.module.app.test_client()
        stolen.set_cookie('session', cookie.value)
        assert stolen.get('/api/status').status_code == 200
        assert sandbox.post('/logout', data={}).status_code == 302
        assert stolen.get('/api/status').status_code == 401


def test_unavailable_access_store_blocks_authenticated_mutations() -> None:
    with LegacySandbox() as sandbox:
        sandbox.seed_price('BTC/USDT', 100.0)
        # Simulate loss of the access-control store at its supported IO boundary.
        (sandbox.directory / 'operator-auth.sqlite3').unlink()
        response = sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'})
        assert response.status_code == 503
        assert sandbox.exchange.holdings == {}
        assert not (sandbox.directory / 'positions.json').exists()


def test_http_fields_cannot_override_venue_credentials_or_execution_gate() -> None:
    with LegacySandbox() as sandbox:
        before = sandbox.client.get('/api/status').get_json()
        sandbox.post('/update-config', data={
            'ALLOW_EXCHANGE_EXECUTION': 'true', 'ALLOW_MAINNET_EXECUTION': 'true',
            'BINANCE_VENUE': 'mainnet', 'API_KEY': 'attacker-key', 'TRADING_MODE': 'exchange',
        })
        after = sandbox.client.get('/api/status').get_json()
        assert after['config'] == before['config']
        assert after['venue'] == before['venue'] == 'testnet'
        assert after['exchange_allowed'] is False
        assert sandbox.exchange.holdings == {}


def test_reset_restores_configured_environment_defaults_and_preserves_mode() -> None:
    with LegacySandbox(environment={'RSI_THRESHOLD': '55', 'USE_DYNAMIC_UNIVERSE': 'false'}) as sandbox:
        before = sandbox.client.get('/api/status').get_json()['config']
        assert before['rsi_threshold'] == 55.0
        assert before['use_dynamic_universe'] is False
        sandbox.post('/update-config', data={'rsi_threshold': '62', 'use_dynamic_universe': 'true'})
        sandbox.post('/reset-config', data={})
        reset = sandbox.client.get('/api/status').get_json()['config']
        assert reset['rsi_threshold'] == 55.0
        assert reset['use_dynamic_universe'] is False
        assert reset['paper_trading'] is True
