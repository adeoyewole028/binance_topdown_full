
from flask import Flask, render_template_string, request, redirect, url_for, flash, jsonify, make_response
import time
import os
import pandas as pd
from utils import ema, rsi, is_higher_highs_lows, detect_bullish_engulfing, detect_hammer
from typing import Mapping
from operator_security import install_operator_security, configure_operator
from runtime_environment import RuntimeEnvironment, construct_exchange, parse_boolean
from state_store import StateStore, StateUnavailable

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
    'balances': {'USDT': 1000, 'BTC': 0.1},
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
def compute_exit_reasons(symbol: str, entry: float, qty: float, exchange) -> dict:
    reasons = []
    try:
        price = exchange.fetch_ticker(symbol)['last']
    except Exception:
        price = status['live_prices'].get(symbol)
    if not isinstance(price, (int, float)) or price is None:
        return {'should_exit': False, 'reasons': [], 'price': None}

    tp_price = entry * (1 + status['config']['take_profit_pct'])
    sl_price = entry * (1 - status['config']['stop_loss_pct'])
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
            if (prev['rsi'] >= status['config']['rsi_overbought']) and (last['rsi'] < status['config']['rsi_overbought']):
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

def initialize_runtime(environment: Mapping[str, str] | None = None) -> Flask:
    """Explicit web startup; no scanner or exchange requests are started here."""
    global runtime_environment, DATA_DIR, configured_defaults, state_store
    environment = os.environ if environment is None else environment
    runtime = RuntimeEnvironment.from_mapping(environment)
    app.config.update(DEBUG=False)
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
    if request.endpoint in {'operator_login', 'logout'} or state_store is None:
        return None
    try:
        status.update(state_store.scan())
        status['config_version'], status['config'] = state_store.config()
        status['open_positions'] = [{'symbol': p['symbol'], 'qty': float(p['qty']), 'entry': float(p['entry']), 'pnl': 0}
                                    for p in state_store.positions()]
    except StateUnavailable:
        return jsonify(error='Durable state unavailable; execution paused'), 503
    return None

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
    symbols = ['ETH/USDT', 'BTC/USDT', 'SOL/USDT', 'BNB/USDT']
    global exchange
    if runtime_environment is None:
        raise RuntimeError('Explicit runtime initialization is required')
    exchange = construct_exchange(runtime_environment, private=False)
    while True:
        from datetime import datetime, timezone
        status['config_version'], status['config'] = state_store.config()
        status['open_positions'] = [{'symbol': p['symbol'], 'qty': float(p['qty']), 'entry': float(p['entry']), 'pnl': 0}
                                    for p in state_store.positions()]
        now = datetime.now(timezone.utc).isoformat()
        status['last_scan'] = now
        # Dynamic universe selection from top quote volume
        if status['config']['use_dynamic_universe']:
            try:
                tickers_all = exchange.fetch_tickers()
                candidates = []
                for sym, data in tickers_all.items():
                    if not sym.endswith('/' + status['config']['base_quote']):
                        continue
                    if any(x in sym for x in ['UP/', 'DOWN/', 'BULL/', 'BEAR/']):
                        continue
                    qvol = data.get('quoteVolume') or 0
                    if qvol >= status['config']['min_quote_vol']:
                        candidates.append((sym, qvol))
                candidates.sort(key=lambda x: x[1], reverse=True)
                symbols = [s for s, _ in candidates[:status['config']['universe_size']]] or symbols
            except Exception as e:
                status['logs'].append(f"Universe error: {type(e).__name__}")
        status['universe'] = symbols
        # Fetch live prices (with retries)
        try:
            def _fetch():
                return exchange.fetch_tickers(symbols)
            tickers = None
            delay = 0.5
            for attempt in range(3):
                try:
                    tickers = _fetch()
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(delay)
                    delay *= 2
            status['live_prices'] = {s: tickers[s]['last'] for s in symbols}
        except Exception as e:
            status['live_prices'] = {s: 'N/A' for s in symbols}
            status['logs'].append(f"Error fetching prices: {type(e).__name__}")
        # Update P&L for local open positions
        for pos in status['open_positions']:
            sym = pos['symbol']
            price = status['live_prices'].get(sym)
            try:
                if isinstance(price, (int, float)):
                    pos['pnl'] = (float(price) - float(pos['entry'])) * float(pos['qty'])
            except Exception:
                pos['pnl'] = 0
    # Balances disabled (no-op)
        # recent_trades retained in memory (trim occasionally)
        if len(status.get('recent_trades', [])) > 50:
            status['recent_trades'] = status['recent_trades'][:50]
        
        # Update performance metrics (live)
        try:
            closed = [t for t in status.get('recent_trades', []) if t.get('side') == 'sell' and t.get('status') == 'filled']
            wins = sum(1 for t in closed if (t.get('pnl') or 0) > 0)
            total = len(closed)
            win_rate = (wins / total) if total else 0.0

            realized = sum(float(t.get('pnl') or 0.0) for t in closed)
            unreal = 0.0
            for pos in status.get('open_positions', []):
                price = status['live_prices'].get(pos['symbol'])
                if isinstance(price, (int, float)):
                    try:
                        unreal += (float(price) - float(pos['entry'])) * float(pos['qty'])
                    except Exception:
                        pass
            total_profit = realized + unreal

            # equity tracking (relative)
            equity = total_profit
            status['equity_history'].append(equity)
            if len(status['equity_history']) > 1000:
                status['equity_history'] = status['equity_history'][-1000:]
            if equity > status['peak_equity']:
                status['peak_equity'] = equity
            peak = status['peak_equity'] or 0.0
            drawdown = ((peak - equity) / peak * 100.0) if peak > 0 else 0.0

            status['performance']['win_rate'] = win_rate
            status['performance']['total_profit'] = round(total_profit, 4)
            status['performance']['drawdown'] = round(drawdown, 2)
        except Exception as e:
            status['logs'].append(f"Perf calc error: {type(e).__name__}")
        # Multi-timeframe signal gating: 4h trend + 1h bias + 15m execution
        detected_signals = []
        for symbol in symbols:
            try:
                df_4h_trend = fetch_ohlcv_df(exchange, symbol, '4h', 200)
                df_1h = fetch_ohlcv_df(exchange, symbol, '1h', 200)
                df_15m = fetch_ohlcv_df(exchange, symbol, '15m', 100)
                # Use the same structure logic on 4h for trend
                trend4h = daily_trend_check(df_4h_trend)
                bias1h = bias_check_1h(df_1h)
                exec15 = execution_check_15m(df_15m)
                if trend4h['bullish'] and bias1h['near_ma'] and bias1h['momentum_ok'] and exec15['signal']:
                    detected_signals.append({
                        'symbol': symbol,
                        'type': exec15.get('type', 'entry'),
                        'time': now,
                        'trend4h': trend4h,
                        'bias1h': bias1h
                    })
                else:
                    status['logs'].append(
                        f"NoSignal {symbol}: 4hTrend={trend4h['bullish']} nearEMA1h={bias1h['near_ma']} RSI1h={bias1h['rsi']:.2f} exec={exec15['signal']}"
                    )
            except Exception as e:
                status['logs'].append(f"Signal error for {symbol}: {type(e).__name__}")
        status['signals'] = detected_signals
        # Simulate cooldowns
        # Build exit signals for open positions
        exit_sigs = []
        for pos in status['open_positions']:
            sym = pos['symbol']
            entry = float(pos['entry'])
            qty = float(pos['qty'])
            res = compute_exit_reasons(sym, entry, qty, exchange)
            if res['should_exit']:
                pnl = (res['price'] - entry) * qty
                exit_sigs.append({'symbol': sym, 'reasons': res['reasons'], 'price': res['price'], 'pnl': pnl})
        status['exit_signals'] = exit_sigs

        # Simulate cooldowns
        status['cooldowns'] = {s: 0 for s in symbols}
        # Logs
        status['logs'].append(f"Scanned at {now}")
        if len(status['logs']) > 20:
            status['logs'] = status['logs'][-20:]
        owner.publish_scan(status)
        owner.process_pending()
        time.sleep(10)

@app.route('/trade', methods=['POST'])
def trade():
    symbol = request.form.get('symbol')
    side = request.form.get('side')
    qty = request.form.get('qty')
    if not status['config']['paper_trading']:
        try:
            require_exchange_execution()
        except ValueError:
            return jsonify(error='Exchange execution is disabled by server policy'), 403
    try:
        operation = state_store.enqueue('trade', symbol, side, qty)
    except StateUnavailable:
        return jsonify(error='Order was not accepted; durable state unavailable'), 503
    flash(f'Order queued for the execution owner: {operation}', 'success')
    return redirect('/')

@app.route('/close', methods=['POST'])
def close_position():
    symbol = request.form.get('symbol')
    qty = request.form.get('qty')
    if not status['config']['paper_trading']:
        try:
            require_exchange_execution()
        except ValueError:
            return jsonify(error='Exchange execution is disabled by server policy'), 403
    # Preserve the canonical decimal boundary; display floats are estimates.
    q = qty
    if not q:
        q = next((p['qty'] for p in state_store.positions() if p['symbol'] == symbol), '0')
    try:
        operation = state_store.enqueue('close', symbol, 'sell', str(q))
    except StateUnavailable:
        return jsonify(error='Close was not accepted; durable state unavailable'), 503
    flash(f'Close queued for the execution owner: {operation}', 'success')
    return redirect('/')

@app.route('/api/status', methods=['GET'])
def api_status():
    # Minimal JSON for static frontends; includes signals, positions, performance, config (non-secret)
    payload = {
        'last_scan': status['last_scan'],
        'universe': status['universe'],
        'signals': status['signals'],
        'exit_signals': status['exit_signals'],
        'open_positions': status['open_positions'],
        'performance': status['performance'],
        'config': status['config'],
        'live_prices': status['live_prices'],
        'execution_mode': 'paper' if status['config']['paper_trading'] else 'exchange',
        'venue': runtime_environment.venue if runtime_environment else 'unconfigured',
        'exchange_allowed': runtime_environment.allow_exchange if runtime_environment else False,
        'operations': state_store.operations(),
        'positions': state_store.positions(),
        'migration': state_store.migration(),
        'ownership': state_store.ownership(),
    }
    resp = make_response(jsonify(payload), 200)
    return resp

@app.route('/update-config', methods=['POST'])
def update_config():
    # Build dict from form fields, then coerce types
    form = request.form.to_dict(flat=True)
    # Absence always means preserve; safety mode is a separate reviewed action.
    updates = {k: v for k, v in form.items() if k in status['config']}
    if 'paper_trading' in updates:
        try:
            if parse_boolean(updates['paper_trading']) != status['config']['paper_trading']:
                return jsonify(error='Use a reviewed execution mode transition'), 409
        except ValueError:
            return jsonify(error='Invalid execution mode'), 400
    coerced = coerce_config_types(updates)
    # Merge into status
    try:
        status['config_version'], status['config'] = state_store.update_config(coerced, 'operator')
        ok = True
    except StateUnavailable:
        return jsonify(error='Durable state unavailable; configuration unchanged'), 503
    if ok:
        flash('Configuration saved.', 'success')
    else:
        flash('Failed to save configuration; using in-memory values.', 'error')
    return redirect(url_for('dashboard'))

@app.route('/reset-config', methods=['POST'])
def reset_config():
    # Reset to DEFAULT_CONFIG and persist
    paper = status['config']['paper_trading']
    try:
        status['config_version'], status['config'] = state_store.update_config({**configured_defaults, 'paper_trading': paper}, 'operator-reset')
        ok = True
    except StateUnavailable:
        return jsonify(error='Durable state unavailable; configuration unchanged'), 503
    if ok:
        flash('Configuration reset to defaults.', 'success')
    else:
        flash('Reset failed to save; defaults applied in memory.', 'error')
    return redirect(url_for('dashboard'))

@app.post('/change-mode')
def change_mode():
    mode = request.form.get('mode')
    if mode not in {'paper', 'exchange'}:
        return jsonify(error='Invalid execution mode'), 400
    current = 'paper' if status['config']['paper_trading'] else 'exchange'
    if request.form.get('expected_mode') != current or request.form.get('confirmation') != mode:
        return jsonify(error='Review and confirm the current execution mode'), 409
    if mode != current:
        # Actual cross-namespace transitions require BT03 durable recovery and
        # account reconciliation. Never reinterpret legacy positions as holdings.
        return jsonify(error='Mode transition requires isolated state and account reconciliation; restart after review'), 409
    return redirect(url_for('dashboard'))

@app.route('/', methods=['GET'])
def dashboard():
    return render_template_string('''
        <!doctype html>
        <html lang="en" class="h-full bg-slate-950">
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <meta http-equiv="refresh" content="15" />
            <title>Binance Topdown Bot Dashboard</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="h-full text-slate-100">
            <header class="bg-gradient-to-r from-slate-900 to-slate-800 border-b border-slate-800">
                <div class="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
                    <h1 class="text-xl font-semibold">Binance Topdown Bot</h1>
                    <span>Mode: {{ 'Paper' if status['config']['paper_trading'] else 'Exchange' }} / {{ venue }}</span>
                    <form method="post" action="/logout">
                        <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
                        <button type="submit">Sign out</button>
                    </form>
                    <div class="flex items-center gap-2">
                        <button id="soundToggle" class="px-2 py-1 rounded border border-slate-700 text-xs text-slate-200 hover:bg-slate-800">Sound: Off</button>
                        <span class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium {{ 'bg-emerald-600 text-slate-900' if status['bot_status']=='Running' else 'bg-rose-600 text-white' }}">
                            {{ status['bot_status'] }}
                        </span>
                    </div>
                </div>
            </header>

            <!-- Toasts root -->
            <div id="toast-root" class="fixed top-4 right-4 z-50 space-y-2"></div>

            <main class="max-w-7xl mx-auto p-4 space-y-4">
                <section class="grid grid-cols-1 xl:grid-cols-4 gap-4">
                    <div class="col-span-1 bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                        <h2 class="text-sm font-semibold text-slate-300">Overview</h2>
                        <div class="text-xs text-slate-400 mt-1">Last scan: {{ status['last_scan'] }}</div>
                        <div class="mt-3 space-y-1 text-sm">
                            <div class="flex items-center justify-between">
                                <span class="text-slate-400">Universe size</span>
                                <span class="px-2 py-0.5 rounded bg-slate-700 text-slate-200 text-xs">{{ status['universe']|length }}</span>
                            </div>
                            <div class="flex items-center justify-between">
                                <span class="text-slate-400">Signals</span>
                                <span class="px-2 py-0.5 rounded {{ 'bg-emerald-600 text-slate-900' if status['signals'] else 'bg-slate-700 text-slate-200' }} text-xs">{{ status['signals']|length }}</span>
                            </div>
                            <div class="flex items-center justify-between">
                                <span class="text-slate-400">Exit signals</span>
                                <span class="px-2 py-0.5 rounded {{ 'bg-amber-500 text-slate-900' if status['exit_signals'] else 'bg-slate-700 text-slate-200' }} text-xs">{{ status['exit_signals']|length }}</span>
                            </div>
                        </div>
                    </div>

                    <div class="col-span-1 xl:col-span-3 bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                        <h2 class="text-sm font-semibold text-slate-300">Universe</h2>
                        <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-3 mt-3">
                            {% for symbol in status['universe'][:12] %}
                            <div class="rounded-lg border border-slate-800 bg-slate-900/60 p-3">
                                <div class="flex items-center justify-between">
                                    <div class="font-semibold text-sm">{{ symbol }}</div>
                                    <span class="px-2 py-0.5 rounded bg-slate-700 text-slate-200 text-xs">{{ status['cooldowns'][symbol] }}s</span>
                                </div>
                                <div class="mt-2 text-sm">
                                    {% set p = status['live_prices'][symbol] %}
                                    {% if p == 'N/A' %}
                                        <span class="italic text-slate-400">N/A</span>
                                    {% else %}
                                        <span class="text-slate-200">{{ p|round(6) }}</span>
                                    {% endif %}
                                </div>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                </section>

                <section class="grid grid-cols-1 gap-4">
                    <div class="bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                        <h2 class="text-sm font-semibold text-slate-300">Open Positions</h2>
                        <div class="overflow-x-auto mt-2">
                            <table class="min-w-full text-sm">
                                <thead class="text-slate-400">
                                    <tr class="border-b border-slate-800">
                                        <th class="text-left py-2">Symbol</th>
                                        <th class="text-left py-2">Entry</th>
                                        <th class="text-left py-2">Qty</th>
                                        <th class="text-left py-2">P&L</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for pos in status['open_positions'] %}
                                    {% set pnlpos = 'text-emerald-400' if pos['pnl']>0 else ('text-rose-400' if pos['pnl']<0 else 'text-slate-200') %}
                                    <tr class="border-b border-slate-800/60">
                                        <td class="py-2">{{ pos['symbol'] }}</td>
                                        <td class="py-2">{{ pos['entry'] }}</td>
                                        <td class="py-2">{{ pos['qty'] }}</td>
                                        <td class="py-2 {{ pnlpos }}">{{ pos['pnl']|round(2) }}</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </section>

                <section class="grid grid-cols-1 xl:grid-cols-3 gap-4">
                    <div class="xl:col-span-2 bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                        <h2 class="text-sm font-semibold text-slate-300">Signals</h2>
                        {% if status['signals'] %}
                        <div class="overflow-x-auto mt-2">
                            <table class="min-w-full text-sm">
                                <thead class="text-slate-400">
                                    <tr class="border-b border-slate-800">
                                        <th class="text-left py-2">Time</th>
                                        <th class="text-left py-2">Symbol</th>
                                        <th class="text-left py-2">Type</th>
                                        <th class="text-left py-2">4h Trend</th>
                                        <th class="text-left py-2">1h Bias</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for sig in status['signals'] %}
                                    <tr class="border-b border-slate-800/60">
                                        <td class="py-2 text-slate-400">{{ sig['time'] }}</td>
                                        <td class="py-2 font-medium">{{ sig['symbol'] }}</td>
                                        <td class="py-2">
                                            <span class="inline-flex items-center rounded-full bg-emerald-600 text-slate-900 px-2 py-0.5 text-xs font-medium">{{ sig['type'] }}</span>
                                        </td>
                                        <td class="py-2 space-x-1">
                                            <span class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium {{ 'bg-emerald-600 text-slate-900' if sig['trend4h']['bullish'] else 'bg-slate-700 text-slate-200' }}">bullish</span>
                                            <span class="inline-flex items-center rounded-full bg-slate-700 text-slate-200 px-2 py-0.5 text-xs font-medium">21/{{ 'Y' if sig['trend4h']['above21'] else 'N' }}</span>
                                            <span class="inline-flex items-center rounded-full bg-slate-700 text-slate-200 px-2 py-0.5 text-xs font-medium">HHLL/{{ 'Y' if sig['trend4h']['hhll'] else 'N' }}</span>
                                        </td>
                                        <td class="py-2 space-x-1">
                                            <span class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium {{ 'bg-emerald-600 text-slate-900' if sig['bias1h']['near_ma'] else 'bg-slate-700 text-slate-200' }}">near EMA21</span>
                                            <span class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium {{ 'bg-emerald-600 text-slate-900' if sig['bias1h']['momentum_ok'] else 'bg-slate-700 text-slate-200' }}">RSI ok</span>
                                            <span class="inline-flex items-center rounded-full bg-slate-700 text-slate-200 px-2 py-0.5 text-xs font-medium">RSI {{ sig['bias1h']['rsi']|round(2) }}</span>
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                        {% else %}
                            <div class="text-slate-400 mt-2">No signals detected on the current scan.</div>
                        {% endif %}
                    </div>

                    <div class="bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                        <h2 class="text-sm font-semibold text-slate-300">Exit Signals</h2>
                        {% if status['exit_signals'] %}
                        <div class="overflow-x-auto mt-2">
                            <table class="min-w-full text-sm">
                                <thead class="text-slate-400">
                                    <tr class="border-b border-slate-800"><th class="text-left py-2">Symbol</th><th class="text-left py-2">Price</th><th class="text-left py-2">P&L</th><th class="text-left py-2">Reasons</th><th class="py-2"></th></tr>
                                </thead>
                                <tbody>
                                    {% for es in status['exit_signals'] %}
                                    {% set pnlpos = 'text-emerald-400' if es['pnl']>0 else ('text-rose-400' if es['pnl']<0 else 'text-slate-200') %}
                                    <tr class="border-b border-slate-800/60">
                                        <td class="py-2">{{ es['symbol'] }}</td>
                                        <td class="py-2">{{ es['price'] }}</td>
                                        <td class="py-2 {{ pnlpos }}">{{ es['pnl']|round(2) }}</td>
                                        <td class="py-2 space-x-1">
                                            {% for r in es['reasons'] %}
                                                <span class="inline-flex items-center rounded-full bg-amber-500 text-slate-900 px-2 py-0.5 text-xs font-medium">{{ r }}</span>
                                            {% endfor %}
                                        </td>
                                        <td class="py-2">
                                            <form method="post" action="/close" class="inline-flex">
                                                <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
                                                <input type="hidden" name="symbol" value="{{ es['symbol'] }}">
                                                <input type="hidden" name="qty" value="{% for pos in status['open_positions'] %}{% if pos['symbol']==es['symbol'] %}{{ pos['qty'] }}{% endif %}{% endfor %}">
                                                <button type="submit" class="px-2 py-1 rounded border border-slate-600 text-slate-200 hover:bg-slate-800 text-xs">Close</button>
                                            </form>
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                        {% else %}
                            <div class="text-slate-400 mt-2">No exit signals right now.</div>
                        {% endif %}
                    </div>
                </section>

                <section class="grid grid-cols-1 xl:grid-cols-3 gap-4">
                    <div class="xl:col-span-2 bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                        <h2 class="text-sm font-semibold text-slate-300">Place Trade</h2>
                                    <form method="post" action="/trade" class="mt-2 grid grid-cols-1 sm:grid-cols-4 gap-2 items-end">
                                        <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
                                        <label class="text-sm">Symbol
                                            {% if status['universe'] %}
                                                <select class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 text-sm" name="symbol">
                                                    {% for s in status['universe'] %}
                                                        <option value="{{ s }}">{{ s }}</option>
                                                    {% endfor %}
                                                </select>
                                            {% else %}
                                                <input class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 text-sm" name="symbol" value="BTC/USDT" />
                                            {% endif %}
                                        </label>
                            <label class="text-sm">Side
                                <select class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 text-sm" name="side">
                                    <option value="buy">Buy</option>
                                    <option value="sell">Sell</option>
                                </select>
                            </label>
                            <label class="text-sm">Quantity
                                <div class="flex gap-2 items-center">
                                    <input class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 text-sm" name="qty" id="qtyInput" type="number" step="any" value="0.01" />
                                    <button type="button" id="riskSizeBtn" class="mt-1 px-2 py-1 rounded border border-slate-600 text-slate-200 hover:bg-slate-800 text-xs">Size by risk</button>
                                </div>
                                <div class="text-xs text-slate-500 mt-1">Uses equity_proxy and stop_loss_pct to estimate qty.</div>
                            </label>
                            <button type="submit" class="w-full sm:w-auto px-3 py-2 rounded bg-indigo-600 hover:bg-indigo-500 text-sm font-medium">Submit</button>
                        </form>
                        <div class="mt-3 space-y-1">
                            {% with messages = get_flashed_messages(with_categories=true) %}
                                {% if messages %}
                                    {% for category, message in messages %}
                                        <div class="px-3 py-2 rounded text-sm {{ 'bg-emerald-600 text-slate-900' if category=='success' else 'bg-rose-600 text-white' }}">{{ message }}</div>
                                    {% endfor %}
                                {% endif %}
                            {% endwith %}
                        </div>
                        <form method="post" action="/reset-config" class="mt-3">
                            <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
                            <button type="submit" class="px-3 py-2 rounded border border-slate-700 text-slate-200 hover:bg-slate-800 text-xs">Reset to defaults</button>
                        </form>
                    </div>

                    <div class="bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                        <h2 class="text-sm font-semibold text-slate-300">Performance</h2>
                        <div class="grid grid-cols-2 gap-4 mt-2">
                            <div>
                                <div class="text-xs text-slate-400">Win Rate</div>
                                <div class="text-lg font-semibold">{{ status['performance']['win_rate']*100|round(2) }}%</div>
                            </div>
                            <div>
                                <div class="text-xs text-slate-400">Total Profit</div>
                                <div class="text-lg font-semibold">{{ status['performance']['total_profit'] }}</div>
                            </div>
                        </div>
                        <div class="mt-3">
                            <div class="text-xs text-slate-400">Drawdown</div>
                            <div class="text-base">{{ status['performance']['drawdown'] }}</div>
                        </div>
                        <div class="mt-4">
                            <h3 class="text-sm font-semibold text-slate-300">Config</h3>
                            <form method="post" action="/update-config" class="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-2 text-sm">
                                <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
                                <label>risk
                                    <input name="risk" type="number" step="0.001" min="0" max="1" value="{{ status['config']['risk'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>max_positions
                                    <input name="max_positions" type="number" step="1" min="1" value="{{ status['config']['max_positions'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>cooldown
                                    <input name="cooldown" type="number" step="1" min="0" value="{{ status['config']['cooldown'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <p>Execution mode: {{ 'Paper' if status['config']['paper_trading'] else 'Exchange' }}.
                                    Mode and venue changes require isolated state and account review.</p>
                                <label>near_ema_pct
                                    <input name="near_ema_pct" type="number" step="0.001" min="0" value="{{ status['config']['near_ema_pct'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>rsi_threshold
                                    <input name="rsi_threshold" type="number" step="1" min="0" max="100" value="{{ status['config']['rsi_threshold'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>rsi_exec_threshold
                                    <input name="rsi_exec_threshold" type="number" step="1" min="0" max="100" value="{{ status['config']['rsi_exec_threshold'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>rsi_overbought
                                    <input name="rsi_overbought" type="number" step="1" min="0" max="100" value="{{ status['config']['rsi_overbought'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>take_profit_pct
                                    <input name="take_profit_pct" type="number" step="0.001" min="0" value="{{ status['config']['take_profit_pct'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>stop_loss_pct
                                    <input name="stop_loss_pct" type="number" step="0.001" min="0" value="{{ status['config']['stop_loss_pct'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>trailing_pct
                                    <input name="trailing_pct" type="number" step="0.001" min="0" value="{{ status['config']['trailing_pct'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>universe_size
                                    <input name="universe_size" type="number" step="1" min="1" value="{{ status['config']['universe_size'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>base_quote
                                    <input name="base_quote" type="text" value="{{ status['config']['base_quote'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label class="flex items-center gap-2">use_dynamic_universe
                                    <select name="use_dynamic_universe" class="mt-1 rounded border-slate-700">
                                        <option value="true" {% if status['config']['use_dynamic_universe'] %}selected{% endif %}>Enabled</option>
                                        <option value="false" {% if not status['config']['use_dynamic_universe'] %}selected{% endif %}>Disabled</option>
                                    </select>
                                </label>
                                <label>min_quote_vol
                                    <input name="min_quote_vol" type="number" step="1000" min="0" value="{{ status['config']['min_quote_vol'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>equity_proxy
                                    <input name="equity_proxy" type="number" step="0.01" min="0" value="{{ status['config']['equity_proxy'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>

                                <div class="sm:col-span-2 flex items-center justify-end gap-2">
                                    <button type="submit" class="px-3 py-2 rounded bg-emerald-600 hover:bg-emerald-500 text-slate-900 font-medium">Save</button>
                                </div>
                            </form>
                        </div>
                    </div>
                </section>

                            <section>
                                <div class="bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                                    <h2 class="text-sm font-semibold text-slate-300">Recent Trades</h2>
                                    <div class="overflow-x-auto mt-2">
                                        <table class="min-w-full text-sm">
                                            <thead class="text-slate-400">
                                                <tr class="border-b border-slate-800">
                                                    <th class="text-left py-2">Time</th>
                                                    <th class="text-left py-2">Symbol</th>
                                                    <th class="text-left py-2">Side</th>
                                                    <th class="text-left py-2">Qty</th>
                                                    <th class="text-left py-2">Price</th>
                                                    <th class="text-left py-2">P&L</th>
                                                    <th class="text-left py-2">Status</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {% for t in status['recent_trades'] %}
                                                    {% set pnlc = 'text-emerald-400' if t['pnl']>0 else ('text-rose-400' if t['pnl']<0 else 'text-slate-200') %}
                                                    <tr class="border-b border-slate-800/60">
                                                        <td class="py-2 text-slate-400">{{ t['time'] }}</td>
                                                        <td class="py-2">{{ t['symbol'] }}</td>
                                                        <td class="py-2"><span class="px-2 py-0.5 rounded {{ 'bg-emerald-600 text-slate-900' if t['side']=='buy' else 'bg-rose-600 text-white' }} text-xs">{{ t['side'] }}</span></td>
                                                        <td class="py-2">{{ t['qty'] }}</td>
                                                        <td class="py-2">{{ t['price'] }}</td>
                                                        <td class="py-2 {{ pnlc }}">{{ t['pnl'] }}</td>
                                                        <td class="py-2 text-slate-300">{{ t['status'] }}</td>
                                                    </tr>
                                                {% endfor %}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </section>

                <section>
                    <div class="bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                        <h2 class="text-sm font-semibold text-slate-300">Logs</h2>
                        <div class="mt-2 max-h-64 overflow-auto rounded border border-slate-800 bg-slate-900 px-3 py-2 font-mono text-xs">
                            {% for log in status['logs'] %}
                                <div>{{ log }}</div>
                            {% endfor %}
                        </div>
                    </div>
                </section>
            </main>
                    <script>
                        // Persist and toggle sound preference
                        let soundEnabled = (localStorage.getItem('soundEnabled') === '1');
                        const toggleBtn = document.getElementById('soundToggle');
                        function refreshSoundLabel(){ if(toggleBtn){ toggleBtn.textContent = 'Sound: ' + (soundEnabled ? 'On' : 'Off'); } }
                        refreshSoundLabel();
                        let audioCtx = null;
                        function ensureAudio(){
                            if (!soundEnabled) return;
                            try { audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)(); } catch(e){}
                        }
                        function playBeep(){
                            if (!soundEnabled) return;
                            ensureAudio();
                            if (!audioCtx) return;
                            const o = audioCtx.createOscillator();
                            const g = audioCtx.createGain();
                            o.type = 'sine';
                            o.frequency.value = 880;
                            g.gain.value = 0.08; // slightly louder
                            o.connect(g); g.connect(audioCtx.destination);
                            o.start();
                            setTimeout(()=>{ o.stop(); }, 250); // a bit longer
                        }
                        if (toggleBtn) {
                            toggleBtn.addEventListener('click', async () => {
                                soundEnabled = !soundEnabled;
                                localStorage.setItem('soundEnabled', soundEnabled ? '1' : '0');
                                refreshSoundLabel();
                                if (soundEnabled) { try { ensureAudio(); if (audioCtx && audioCtx.state !== 'running') { await audioCtx.resume(); } playBeep(); } catch(e){} }
                            });
                        }

                        // Re-arm audio on first user interaction after each page load (autoplay policy)
                        if (soundEnabled) {
                            const onFirstInput = async () => {
                                try {
                                    ensureAudio();
                                    if (audioCtx && audioCtx.state !== 'running') { await audioCtx.resume(); }
                                    playBeep();
                                } catch(e) {}
                                window.removeEventListener('pointerdown', onFirstInput);
                            };
                            window.addEventListener('pointerdown', onFirstInput, { once: true });
                        }

                        // Toast helper
                        function showToast(message){
                            const root = document.getElementById('toast-root');
                            if(!root) return;
                            const el = document.createElement('div');
                            el.className = 'rounded border border-slate-700 bg-slate-900/90 text-slate-100 px-3 py-2 shadow-lg text-sm transform transition-all duration-300 translate-y-[-6px] opacity-0';
                            el.innerHTML = message;
                            root.appendChild(el);
                            requestAnimationFrame(()=>{ el.classList.remove('translate-y-[-6px]','opacity-0'); el.classList.add('translate-y-0','opacity-100'); });
                            setTimeout(()=>{
                                el.classList.add('opacity-0');
                                setTimeout(()=>{ el.remove(); }, 300);
                            }, 6000);
                        }

                        // Diff signals vs previous to raise toasts
                        const currentSignals = {{ status['signals'] | tojson }};
                        const ids = currentSignals.map(s => `${s.symbol}|${s.time}|${s.type || 'entry'}`);
                        let prev = [];
                        try { prev = JSON.parse(localStorage.getItem('signalIds') || '[]'); } catch(e) { prev = []; }
                        const newOnes = ids.filter(id => !prev.includes(id));
                        if (newOnes.length > 0) {
                            // Show each as toast and optional beep
                            for (const id of newOnes) {
                                const parts = id.split('|');
                                const sym = parts[0], t = parts[1], typ = parts[2];
                                showToast(`<span class="font-semibold">Signal</span> â€” ${sym} <span class="text-slate-400">(${typ})</span><br><span class="text-slate-400 text-xs">${t}</span>`);
                            }
                            playBeep();
                        }
                        // Update stored ids
                        try { localStorage.setItem('signalIds', JSON.stringify(ids)); } catch(e){}
                    </script>
                    <script>
                        // Risk-based sizing: qty = (equity * risk) / (price * stop_loss_pct)
                        function getSelectedSymbol(){
                            const sel = document.querySelector('select[name="symbol"]');
                            return sel ? sel.value : 'BTC/USDT';
                        }
                        function getLivePrice(sym){
                            try { const lp = {{ status['live_prices'] | tojson }}; return lp[sym]; } catch(e){ return null; }
                        }
                        const riskBtn = document.getElementById('riskSizeBtn');
                        if (riskBtn) {
                            riskBtn.addEventListener('click', ()=>{
                                const sym = getSelectedSymbol();
                                const price = parseFloat(getLivePrice(sym));
                                const equity = parseFloat({{ status['config']['equity_proxy'] }});
                                const risk = parseFloat({{ status['config']['risk'] }});
                                const sl = parseFloat({{ status['config']['stop_loss_pct'] }});
                                if (!isFinite(price) || price <= 0) return;
                                const qty = (equity * risk) / Math.max(price * Math.max(sl, 1e-6), 1e-6);
                                const inp = document.getElementById('qtyInput');
                                if (inp) inp.value = (Math.max(qty, 0)).toFixed(6);
                            });
                        }
                    </script>
        </body>
        </html>
    ''', status=status, venue=runtime_environment.venue if runtime_environment else 'unconfigured')

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
