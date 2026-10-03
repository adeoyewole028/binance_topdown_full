import secrets
from itsdangerous import URLSafeSerializer, BadSignature

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, make_response, g
import time
import os
import pandas as pd
from utils import ema, rsi, is_higher_highs_lows, detect_bullish_engulfing, detect_hammer
from typing import Mapping
from strategy_profiles import PROFILES, evaluate_observation as evaluate_market_snapshot
from operator_security import install_operator_security, configure_operator
from runtime_environment import RuntimeEnvironment, construct_exchange, parse_boolean
from state_store import StateStore, StateUnavailable
from configuration import ConfigurationError, ConfigurationConflict


def order_review_token():
    observed = g.operator_status
    return URLSafeSerializer(app.secret_key, salt='order-review-v1').dumps(state_store.review_context(observed))


def submitted_order_review():
    try:
        value = URLSafeSerializer(app.secret_key, salt='order-review-v1').loads(request.form.get('order_review', ''))
        if not isinstance(value, dict):
            raise BadSignature('Invalid review')
        return value
    except BadSignature:
        raise ConfigurationConflict('Refresh the order price and settings before submitting') from None

# --- Strategy helpers (mirrors topdown_daemon) ---
def daily_trend_check(df_daily: pd.DataFrame):
    df = df_daily.copy()
    df['ema21'] = ema(df['close'], 21)
    df['ema50'] = ema(df['close'], 50)
    above_ma = bool((df['close'].iloc[-1] > df['ema21'].iloc[-1]) and (df['close'].iloc[-1] > df['ema50'].iloc[-1]))
    hhll = bool(is_higher_highs_lows(df['close']))
    return {'bullish': bool(above_ma and hhll), 'above21': above_ma, 'hhll': hhll}

def bias_check_4h(df_4h: pd.DataFrame):
    df = df_4h.copy()
    df['ema21'] = ema(df['close'], 21)
    df['rsi'] = rsi(df['close'], 14)
    last = df.iloc[-1]
    near_ma = bool(abs(last['close'] - df['ema21'].iloc[-1]) / max(float(last['close']), 1e-12) <= status['config']['near_ema_pct'])
    momentum_ok = bool(last['rsi'] > status['config']['rsi_threshold'])
    return {'near_ma': near_ma, 'momentum_ok': momentum_ok, 'rsi': float(last['rsi'])}

def bias_check_1h(df_1h: pd.DataFrame):
    df = df_1h.copy()
    df['ema21'] = ema(df['close'], 21)
    df['rsi'] = rsi(df['close'], 14)
    last = df.iloc[-1]
    near_ma = bool(abs(last['close'] - df['ema21'].iloc[-1]) / max(float(last['close']), 1e-12) <= status['config']['near_ema_pct'])
    momentum_ok = bool(last['rsi'] > status['config']['rsi_threshold'])
    return {'near_ma': near_ma, 'momentum_ok': momentum_ok, 'rsi': float(last['rsi'])}

def execution_check_15m(df_15m: pd.DataFrame):
    # Candlestick entries
    if detect_bullish_engulfing(df_15m):
        return {'signal': True, 'type': 'bullish_engulfing'}
    if detect_hammer(df_15m):
        return {'signal': True, 'type': 'hammer'}

    # Indicator-based entry
    df = df_15m.copy()
    df['ema21'] = ema(df['close'], 21)
    df['rsi'] = rsi(df['close'], 14)
    if len(df) >= 2:
        last, prev = df.iloc[-1], df.iloc[-2]
        price_above = bool(last['close'] > last['ema21'])
        rsi_cross = bool((prev['rsi'] <= status['config']['rsi_exec_threshold']) and (last['rsi'] > status['config']['rsi_exec_threshold']))
        if price_above and rsi_cross:
            return {'signal': True, 'type': 'ema21_rsi_cross'}
    return {'signal': False}

def fetch_ohlcv_df(exchange, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
    delay = 0.5
    last_err = None
    for attempt in range(3):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            return df
        except Exception as e:
            last_err = e
            if attempt == 2:
                break
            time.sleep(delay)
            delay *= 2
    raise last_err

app = Flask(__name__)
install_operator_security(app)

# Defaults for config (used for reset)
DEFAULT_CONFIG = {
    'strategy_profile': 'dashboard_intraday',
    'strategy_version': 1,
    'risk': 0.01,
    'max_positions': 5,
    'cooldown': 120,
    'paper_trading': True,
    'near_ema_pct': 0.02,
    'rsi_threshold': 30.0,
    'rsi_exec_threshold': 45.0,
    'rsi_overbought': 70.0,
    'take_profit_pct': 0.06,
    'stop_loss_pct': 0.03,
    'trailing_pct': 0.03,
    'universe_size': 12,
    'base_quote': 'USDT',
    'use_dynamic_universe': True,
    'min_quote_vol': 10000000.0,
    # proxy account equity for risk sizing
    'equity_proxy': 1000.0,
}

status = {
    'last_scan': '',
    'universe': [],
    'logs': [],
    'open_positions': [],
    'balances': {},
    'recent_trades': [],
    'signals': [],
    'exit_signals': [],
    'bot_status': 'Running',
    'config': DEFAULT_CONFIG.copy(),
    'performance': {
        'win_rate': 0.0,
        'total_profit': 0.0,
        'drawdown': 0.0
    },
    'live_prices': {},
    'cooldowns': {},
    'equity_history': [],
    'peak_equity': 0.0
}

# Configuration inputs; durable versions live in StateStore.
DATA_DIR = os.getenv('DATA_DIR', os.path.dirname(__file__))



def coerce_config_types(incoming: dict) -> dict:
    # Keep a type map for safe casting
    type_map = {
        'risk': float,
        'max_positions': int,
        'cooldown': int,
        'paper_trading': bool,
        'near_ema_pct': float,
        'rsi_threshold': float,
        'rsi_exec_threshold': float,
        'rsi_overbought': float,
        'take_profit_pct': float,
        'stop_loss_pct': float,
        'trailing_pct': float,
        'universe_size': int,
        'base_quote': str,
        'use_dynamic_universe': bool,
        'min_quote_vol': float,
    }
    out = {}
    for k, caster in type_map.items():
        if k not in incoming:
            continue
        v = incoming[k]
        try:
            if caster is bool:
                # Accept truthy strings or existing bool
                if isinstance(v, str):
                    out[k] = parse_boolean(v)
                else:
                    out[k] = bool(v)
            else:
                out[k] = caster(v)
        except Exception:
            # If casting fails, skip to preserve old value
            continue
    return out

# --- Exit strategy helpers ---
def compute_exit_reasons(symbol: str, entry: float, qty: float, exchange, config=None) -> dict:
    settings = status['config'] if config is None else config
    reasons = []
    try:
        price = exchange.fetch_ticker(symbol)['last']
    except Exception:
        price = status['live_prices'].get(symbol)
    if not isinstance(price, (int, float)) or price is None:
        return {'should_exit': False, 'reasons': [], 'price': None}

    tp_price = entry * (1 + settings['take_profit_pct'])
    sl_price = entry * (1 - settings['stop_loss_pct'])
    if price >= tp_price:
        reasons.append('take_profit')
    if price <= sl_price:
        reasons.append('stop_loss')

    # Indicator-based exits on 15m
    try:
        df15 = fetch_ohlcv_df(exchange, symbol, '15m', 100)
        df15['ema21'] = ema(df15['close'], 21)
        df15['rsi'] = rsi(df15['close'], 14)
        if len(df15) >= 2:
            last, prev = df15.iloc[-1], df15.iloc[-2]
            if (prev['close'] >= prev['ema21']) and (last['close'] < last['ema21']):
                reasons.append('ema21_cross_down_15m')
            if (prev['rsi'] >= settings['rsi_overbought']) and (last['rsi'] < settings['rsi_overbought']):
                reasons.append('rsi_overbought_cross_down_15m')
    except Exception as e:
        status['logs'].append(f"Exit check 15m error {symbol}: {type(e).__name__}")

    # 1h momentum weakening
    try:
        df1h = fetch_ohlcv_df(exchange, symbol, '1h', 100)
        df1h['ema21'] = ema(df1h['close'], 21)
        df1h['rsi'] = rsi(df1h['close'], 14)
        if len(df1h) >= 2:
            l1, p1 = df1h.iloc[-1], df1h.iloc[-2]
            # 1h close below EMA21 with RSI falling
            if (p1['close'] >= p1['ema21']) and (l1['close'] < l1['ema21']) and (l1['rsi'] < p1['rsi']):
                reasons.append('ema21_break_1h_rsi_down')
    except Exception as e:
        status['logs'].append(f"Exit check 1h error {symbol}: {type(e).__name__}")

    return {'should_exit': len(reasons) > 0, 'reasons': reasons, 'price': price}


exchange = None

runtime_environment = None
configured_defaults = DEFAULT_CONFIG.copy()
state_store = None

def environment_defaults(environment: Mapping[str, str]) -> dict:
    defaults = DEFAULT_CONFIG.copy()
    for key, current in defaults.items():
        env_name = 'MIN_24H_QUOTE_VOLUME' if key == 'min_quote_vol' else key.upper()
        if key != 'paper_trading' and environment.get(env_name):
            value = environment[env_name]
            defaults[key] = parse_boolean(value) if isinstance(current, bool) else type(current)(value)
    return defaults

def initialize_runtime(environment: Mapping[str, str] | None = None, *, clock=time.time) -> Flask:
    """Explicit web startup; no scanner or exchange requests are started here."""
    global runtime_environment, DATA_DIR, configured_defaults, state_store
    environment = os.environ if environment is None else environment
    runtime = RuntimeEnvironment.from_mapping(environment)
    app.config.update(DEBUG=False, MARKET_CLOCK=clock)
    configure_operator(app, environment.get('OPERATOR_USERNAME', ''),
                       environment.get('OPERATOR_PASSWORD_HASH', ''),
                       environment.get('FLASK_SECRET_KEY', ''), runtime.data_directory,
                       development=runtime.development)
    DATA_DIR = str(runtime.data_directory)
    # Read strategy knobs only after the explicit launcher has loaded .env.
    defaults = environment_defaults(environment)
    configured_defaults = defaults.copy()
    state_store = StateStore(runtime.data_directory, environment.get('LOCAL_ACCOUNT_ID', 'operator-account'), runtime.venue, runtime.mode)
    state_store.validate()
    status['config_version'], status['config'] = state_store.config()
    # Old persisted flags can never elevate execution mode at startup.
    status['config']['paper_trading'] = runtime.mode == 'paper'
    status['open_positions'] = []
    if runtime.mode == 'exchange' and status['open_positions']:
        raise ValueError('Legacy exposure requires review before exchange startup')
    status['bot_status'] = 'Scanner stopped'
    runtime_environment = runtime
    return app

@app.before_request
def refresh_durable_state():
    if request.endpoint in {'operator_login', 'logout','static'} or state_store is None:
        return None
    try:
        snapshot = state_store.snapshot(now=app.config['MARKET_CLOCK']())
        g.operator_status = {**status, **snapshot.pop('scan'), **snapshot}
        g.operator_status['open_positions'] = display_positions(snapshot['positions'])
        from market_runtime import observed_health
        g.operator_status['market_health'] = observed_health(g.operator_status, app.config['MARKET_CLOCK']())
        health = g.operator_status['market_health']
        g.operator_status['bot_status'] = 'Scanner ready' if health['execution_ready'] else 'Scanner paused: ' + str(health['reason'])
        from datetime import datetime
        from operations_policy import OperationsPolicy
        for quote in g.operator_status['market_prices'].values():
            try:
                stamp = datetime.fromisoformat(quote['fetched_at'])
                quote['age_seconds'] = app.config['MARKET_CLOCK']() - stamp.timestamp()
                quote['fresh'] = stamp.tzinfo is not None and 0 <= quote['age_seconds'] < OperationsPolicy().price_expiry
            except (ValueError, TypeError, OverflowError):
                quote['fresh'] = False
        if g.operator_status['market_prices']:
            g.operator_status['live_prices'] = {s: float(q['price']) if q['fresh'] else 'N/A' for s, q in g.operator_status['market_prices'].items()}
    except StateUnavailable:
        return jsonify(error='Durable state unavailable; execution paused'), 503
    return None


@app.after_request
def dashboard_asset_type(response):
    # Windows registry MIME mappings may report .js as text/plain. Keep
    # nosniff protection and explicitly serve our bundled script as JavaScript.
    if request.endpoint=='static' and request.view_args.get('filename')=='dashboard.js':
        response.headers['Content-Type']='text/javascript; charset=utf-8'
    return response

def display_positions(positions):
    return [{'symbol': p['symbol'], 'qty': float(p['qty']),
             'entry': None if p['basis_status'] == 'provider_basis_unknown' else float(p['entry']),
             'pnl': None if p['basis_status'] == 'provider_basis_unknown' else 0} for p in positions]

def require_exchange_execution():
    if runtime_environment is None:
        raise ValueError('Exchange execution is disabled until explicit initialization')
    runtime_environment.require_exchange()
    if runtime_environment.mode != 'exchange':
        raise ValueError('Restart with a reviewed exchange namespace before execution')


def bot_loop(owner=None) -> None:
    if owner is None:
        from execution_worker import ExecutionWorker
        with ExecutionWorker(state_store) as owned_worker:
            return bot_loop(owned_worker)
    if runtime_environment is None:
        raise RuntimeError('Explicit runtime initialization is required')
    from market_adapter import BinancePublicMarket
    from market_runtime import MarketRuntime
    provider = BinancePublicMarket(construct_exchange(runtime_environment, private=False))
    scanner = MarketRuntime(owner, provider, clock=owner.clock, evaluate=evaluate_market_snapshot,
                            reconcile=owner.poll_exchange if owner.exchange_execution is not None else None,
                            whitelist={s.strip() for s in os.getenv('WHITELIST', '').split(',') if s.strip()},
                            blacklist={s.strip() for s in os.getenv('BLACKLIST', '').split(',') if s.strip()})
    while True:
        scanner.step()
        time.sleep(1)

@app.route('/paper/setup', methods=['POST'])
def setup_paper():
    try:
        expected=int(request.form['expected_config_version'])
        state_store.setup_paper(request.form.to_dict(), 'configured-operator',
                                expected_version=expected)
    except ConfigurationConflict as error:
        return settings_error(str(error),409,request.form.to_dict(),{'expected_config_version':'Review the saved version and account before retrying.'})
    except (ConfigurationError, ValueError, KeyError) as error:
        return settings_error(str(error),422,request.form.to_dict(),getattr(error,'fields',{'expected_config_version':'Review the setup values and saved version.'}))
    except StateUnavailable:
        return jsonify(error='Paper setup was not saved; durable state unavailable'), 503
    return settings_saved(expected+1,'Paper assumptions and explicit initial capital recorded')

@app.route('/paper/cancel', methods=['POST'])
def cancel_paper():
    try:
        state_store.cancel_paper(request.form.get('operation_id', ''), 'configured-operator')
    except ConfigurationConflict as error:
        return jsonify(error=str(error)), 409
    except ConfigurationError as error:
        return jsonify(error=str(error)), 400
    except StateUnavailable:
        return jsonify(error='Cancellation was not saved; durable state unavailable'), 503
    flash('Pending paper order canceled; reservation released', 'info')
    return redirect('/')

@app.route('/exchange/setup', methods=['POST'])
def setup_exchange():
    try:
        expected=int(request.form['expected_config_version'])
        state_store.setup_execution(dict(request.form), 'configured-operator', expected_version=expected)
    except ConfigurationConflict as error:
        return settings_error(str(error),409,request.form.to_dict(),{'expected_config_version':'Review the saved version and account before retrying.'})
    except (ConfigurationError, ValueError, KeyError) as error:
        return settings_error(str(error),422,request.form.to_dict(),getattr(error,'fields',{'expected_config_version':'Review the setup values and saved version.'}))
    except StateUnavailable:
        return jsonify(error='Exchange policy was not saved'), 503
    return settings_saved(expected+1,'Exchange risk policy saved; provider reconciliation is still required')

@app.route('/exchange/entries', methods=['POST'])
def exchange_entries():
    action = request.form.get('action')
    if action not in {'pause', 'resume'}:
        return jsonify(error='Choose pause or resume for new entries'), 400
    try:
        state_store.set_entry_pause(action == 'pause', actor='configured-operator')
    except ConfigurationError as error:
        return jsonify(error=str(error)), 400
    except StateUnavailable:
        return jsonify(error='Entry control was not saved; durable state unavailable'), 503
    flash('New entries paused; reductions and reconciliation remain available' if action == 'pause' else 'Entry pause cleared; all risk and reconciliation gates still apply', 'info')
    return redirect('/')


@app.route('/trade', methods=['POST'])
def trade():
    symbol = request.form.get('symbol')
    side = request.form.get('side')
    qty = request.form.get('qty')
    if not g.operator_status['config']['paper_trading']:
        try:
            require_exchange_execution()
        except ValueError:
            return jsonify(error='Exchange execution is disabled by server policy'), 403
    try:
        operation = state_store.enqueue('trade', symbol, side, qty, sizing=request.form.get('sizing', 'quantity'), idempotency_key=request.form.get('idempotency_key') or request.headers.get('Idempotency-Key'), now=app.config['MARKET_CLOCK'](), review=submitted_order_review())
    except ConfigurationConflict as error:
        return jsonify(error=str(error)), 409
    except ConfigurationError as error:
        return jsonify(error=str(error)), 400
    except StateUnavailable:
        return jsonify(error='Order was not accepted; durable state unavailable'), 503
    observed = next(o for o in state_store.operations() if o['operation_id'] == operation)
    flash(f"Order rejected: {operator_label(observed['error_class'])}" if observed['state'] == 'rejected' else f'Order queued: {operation}',
          'error' if observed['state'] == 'rejected' else 'info')
    return redirect('/')

@app.route('/close', methods=['POST'])
def close_position():
    symbol = request.form.get('symbol')
    qty = request.form.get('qty')
    if not g.operator_status['config']['paper_trading']:
        try:
            require_exchange_execution()
        except ValueError:
            return jsonify(error='Exchange execution is disabled by server policy'), 403
    # Preserve the canonical decimal boundary; display floats are estimates.
    q = qty or ''
    try:
        operation = state_store.enqueue('close', symbol, 'sell', str(q), sizing='quantity' if q else 'full_close', idempotency_key=request.form.get('idempotency_key') or request.headers.get('Idempotency-Key'), now=app.config['MARKET_CLOCK'](), review=submitted_order_review())
    except ConfigurationConflict as error:
        return jsonify(error=str(error)), 409
    except ConfigurationError as error:
        return jsonify(error=str(error)), 400
    except StateUnavailable:
        return jsonify(error='Close was not accepted; durable state unavailable'), 503
    observed = next(o for o in state_store.operations() if o['operation_id'] == operation)
    flash(f"Close rejected: {operator_label(observed['error_class'])}" if observed['state'] == 'rejected' else f'Close queued: {operation}',
          'error' if observed['state'] == 'rejected' else 'info')
    return redirect('/')

@app.route('/api/status', methods=['GET'])
def api_status():
    # Minimal JSON for static frontends; includes signals, positions, performance, config (non-secret)
    observed = g.operator_status
    payload = {
        'namespace':state_store.namespace,
        'presentation':operator_presentation(observed),
        'order_review': order_review_token(),
        'last_scan': observed['last_scan'],
        'universe': observed['universe'],
        'signals': observed['signals'],
        'exit_signals': observed['exit_signals'],
        'open_positions': observed['open_positions'],
        'performance': observed['performance'],
        'config': observed['config'],
        'config_version': observed['config_version'],
        'strategy_decisions': observed.get('strategy_decisions',[]),
        'strategy_profile': observed['config']['strategy_profile'],
        'strategy_version': observed['config']['strategy_version'],
        'trailing_supported': False,
        'market_health': observed['market_health'],
        'market_prices': observed['market_prices'],
        'market_budget': observed['market_budget'],
        'position_rules': {s: {k: v for k, v in rule.items() if k != 'config'} for s, rule in observed['position_rules'].items()},
        'live_prices': observed['live_prices'],
        'execution_mode': 'paper' if observed['config']['paper_trading'] else 'exchange',
        'venue': runtime_environment.venue if runtime_environment else 'unconfigured',
        'exchange_allowed': runtime_environment.allow_exchange if runtime_environment else False,
        'automation': observed['automation'],
        'operations': observed['operations'],
        'positions': observed['positions'],
        'migration': observed['migration'],
        'ownership': observed['ownership'],
        'paper_account': observed['paper_account'],
        'exchange_account': observed['exchange_account'],
        'entry_control': observed['entry_control'],
        'exchange_risk': observed['exchange_risk'],
        'fills': observed['fills'],
        'protection': observed['protection'],
    }
    if request.args.get('view')=='1':payload['dashboard_view']=render_dashboard()
    resp = make_response(jsonify(payload), 200)
    return resp

@app.route('/api/performance', methods=['GET'])
def api_performance():
    try:
        offset=int(request.args.get('offset','0'))
        limit=int(request.args.get('limit','20'))
        if offset<0 or not 1<=limit<=1000:raise ValueError('Invalid history page')
    except (ValueError,TypeError):
        return jsonify(error='Offset must be nonnegative and limit between 1 and 1000'),422
    observed=g.operator_status
    return jsonify(performance=observed['performance'],fills=observed['fills'][offset:offset+limit],
                   history_total=len(observed['fills']),offset=offset,limit=limit)

@app.route('/update-config', methods=['POST'])
def update_config():
    # Build dict from form fields, then coerce types
    form = request.form.to_dict(flat=True)
    # Absence always means preserve; safety mode is a separate reviewed action.
    updates = {k: v for k, v in form.items() if k in g.operator_status['config']}
    if 'paper_trading' in updates:
        try:
            if parse_boolean(updates['paper_trading']) != g.operator_status['config']['paper_trading']:
                return jsonify(error='Use a reviewed execution mode transition'), 409
        except ValueError:
            return jsonify(error='Invalid execution mode'), 400
    # Validate the complete merged version within the write transaction.
    # Merge into status
    try:
        expected = int(form.get('expected_config_version', g.operator_status['config_version']))
        version,_=state_store.update_config(updates, 'operator', expected_version=expected)
    except ConfigurationConflict as error:
        return settings_error(str(error),409,form,{'expected_config_version':'Settings changed. Review the current version before saving again.'})
    except (ConfigurationError, ValueError) as error:
        return settings_error(str(error),422,form,invalid_config_fields(updates))
    except StateUnavailable:
        return jsonify(error='Durable state unavailable; configuration unchanged'), 503
    return settings_saved(version,'Configuration saved.')

@app.route('/reset-config', methods=['POST'])
def reset_config():
    # Reset to DEFAULT_CONFIG and persist
    paper = g.operator_status['config']['paper_trading']
    try:
        expected = int(request.form.get('expected_config_version', g.operator_status['config_version']))
        version,_=state_store.update_config({**configured_defaults, 'paper_trading': paper}, 'operator-reset', expected_version=expected)
    except ConfigurationConflict as error:
        return jsonify(error=str(error)), 409
    except (ConfigurationError, ValueError) as error:
        return jsonify(error=str(error)), 400
    except StateUnavailable:
        return jsonify(error='Durable state unavailable; configuration unchanged'), 503
    return settings_saved(version,'Configuration reset to defaults.')

@app.post('/change-mode')
def change_mode():
    mode = request.form.get('mode')
    if mode not in {'paper', 'exchange'}:
        return jsonify(error='Invalid execution mode'), 400
    current = 'paper' if g.operator_status['config']['paper_trading'] else 'exchange'
    if request.form.get('expected_mode') != current or request.form.get('confirmation') != mode:
        return jsonify(error='Review and confirm the current execution mode'), 409
    if mode != current:
        # Actual cross-namespace transitions require BT03 durable recovery and
        # account reconciliation. Never reinterpret legacy positions as holdings.
        return jsonify(error='Mode transition requires isolated state and account reconciliation; restart after review'), 409
    return redirect(url_for('dashboard'))

@app.route('/', methods=['GET'])
def dashboard():
    return render_dashboard()


def operator_presentation(observed):
    health=observed['market_health']
    reason=health.get('reason')
    ready=health['execution_ready']
    state='ready' if ready else 'unavailable' if not health.get('last_completed_scan') else 'degraded'
    label='Market data ready' if ready else 'Market data unavailable' if state=='unavailable' else 'Market data degraded'
    if ready and not observed['signals']:
        state='no_signal'
        label='Market data ready; no entry signal'
    explanations={'scanner_uninitialized':'Start the supervised scanner to obtain market data.',
        'scanner_stale':'The last successful scan is too old.', 'owner_unavailable':'The execution owner is unavailable.',
        'configuration_changed':'The scanner must observe the current settings.',
        'candle_stale':'Completed candle data is stale.', 'price_stale':'Prices are stale.',
        'market_budget_unavailable':'The shared market request budget is unavailable.'}
    return {'state':state,'label':label,'detail':explanations.get(reason,'Market data needs review.' if reason else ''),
        'scan_age_seconds':health.get('scan_age_seconds'),'last_completed_scan':health.get('last_completed_scan'),
        'permission': 'Exchange execution permitted by server' if runtime_environment and runtime_environment.allow_exchange else 'Paper only; exchange execution is not permitted'}


def invalid_config_fields(updates):
    from configuration import validated_config
    errors={}
    for field,value in updates.items():
        try:validated_config({**g.operator_status['config'],field:value})
        except ConfigurationError as error:errors[field]=str(error)
    return errors or {field:'Review this value and related settings.' for field in updates}


def settings_error(message,code,submitted,errors):
    if request.accept_mimetypes.best=='text/html':
        return render_dashboard(submitted=submitted,errors=errors,message=message,setup=request.path.endswith('/setup')),code
    return jsonify(error=message,fields=errors,current_config_version=g.operator_status['config_version']),code


def settings_saved(version,message):
    if request.accept_mimetypes.best=='application/json':
        return jsonify(saved_config_version=version,message=message)
    flash(message,'success')
    return redirect(url_for('dashboard'))

def operator_label(value):
    labels = {
        None: 'None', 'none': 'None', 'setup_required': 'Supply every required setup value before trading.',
        'drawdown_limit': 'Drawdown stop reached. New entries are blocked; review account risk.',
        'insufficient_paper_assets': 'Insufficient paper balance. Reduce the quantity or review available funds.',
        'entry_halted': 'New entries are stopped. Review account risk before continuing.',
        'valuation_stale': 'Prices are too old. Wait for fresh market data.',
        'valuation_unavailable': 'Account value is unavailable. Review prices and holdings.',
        'reconciliation_required': 'Wait for the exchange account to be reconciled.',
        'review_required': 'Account review required before new entries.',
        'holding_changed_review_required': 'The holding changed. Review its current quantity before retrying.',
        'dispatch_unknown': 'Order outcome is unknown. Wait for reconciliation; do not resubmit.',
        'uninitialized': 'Account has not been reconciled.', 'ready': 'Ready',
        'queued': 'Queued for the execution owner', 'filled': 'Filled', 'rejected': 'Rejected',
        'waiting_for_setup': 'Waiting for explicit setup', 'waiting_for_market_data': 'Waiting for fresh market data',
        'dispatching': 'Being submitted', 'unknown': 'Outcome unknown; reconciliation required',
        'processing': 'Being processed', 'canceling': 'Cancellation pending', 'canceled': 'Canceled',
        'partially_filled': 'Partly filled', 'active': 'Protection active', 'closed': 'Closed',
        'installing': 'Protection being installed', 'failed': 'Protection failed; review the holding',
        'exchange_confirmed': 'Confirmed by the exchange', 'exchange_unverified': 'Awaiting exchange confirmation',
        'simulation': 'Local paper simulation', 'unverified': 'Awaiting verification',
        'stop_market_oco': 'Stop market with linked take profit',
    }
    if value in labels: return labels[value]
    if isinstance(value,str) and '_' not in value: return value
    return 'Status needs operator review. Check the account and operation details before continuing.'


def render_dashboard(*, submitted=None, errors=None, message=None, setup=False):
    return render_template('dashboard.html', submitted={} if setup else submitted or {}, errors={} if setup else errors or {},
        setup_submitted=(submitted or {}) if setup else {}, setup_errors=(errors or {}) if setup else {}, message=message,
        namespace=state_store.namespace,presentation=operator_presentation(g.operator_status), status=g.operator_status,
        operator_label=operator_label, strategy_profiles=PROFILES, venue=runtime_environment.venue if runtime_environment else 'unconfigured',
        order_key=secrets.token_urlsafe(24), order_review=order_review_token())

if __name__ == '__main__':
    import argparse
    from runtime_environment import load_launcher_environment
    parser = argparse.ArgumentParser(description='Isolated development operator server')
    parser.add_argument('--scan', action='store_true', help='Retired: use run_worker.py --scan')
    arguments = parser.parse_args()
    load_launcher_environment()
    initialize_runtime()
    if not runtime_environment.development:
        raise ValueError('Use the production WSGI launcher for production')
    if arguments.scan:
        raise ValueError('Launch the separately owned scanner with python run_worker.py --scan')
    app.run(host='127.0.0.1', port=int(os.getenv('PORT', 5000)), debug=False, use_reloader=False)
