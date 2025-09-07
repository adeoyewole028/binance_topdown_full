#!/usr/bin/env python3
import os, signal, time, math
from datetime import datetime, timedelta
import ccxt, pandas as pd
from utils import ema, rsi, is_higher_highs_lows, detect_bullish_engulfing, detect_hammer, log_trade_csv

running = True
def signal_handler(sig, frame):
    global running
    print(f"Received signal {sig}, shutting down...")
    running = False
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def make_exchange(testnet=True):
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'},
                             'apiKey': os.getenv('API_KEY'), 'secret': os.getenv('API_SECRET')})
    if testnet:
        exchange.set_sandbox_mode(True)
        exchange.urls['api']['public'] = 'https://testnet.binance.vision/api'
        exchange.urls['api']['private'] = 'https://testnet.binance.vision/api'
    return exchange

def fetch_ohlcv_df(exchange, symbol, timeframe, limit=500):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df

def daily_trend_check(df_daily):
    df = df_daily.copy()
    df['ema21'] = ema(df['close'], 21)
    df['ema50'] = ema(df['close'], 50)
    above_ma = df['close'].iloc[-1] > df['ema21'].iloc[-1] and df['close'].iloc[-1] > df['ema50'].iloc[-1]
    hhll = is_higher_highs_lows(df['close'])
    return {'bullish': above_ma and hhll, 'above21': above_ma, 'hhll': hhll}

def bias_check_4h(df_4h):
    df = df_4h.copy()
    df['ema21'] = ema(df['close'], 21)
    df['rsi'] = rsi(df['close'], 14)
    last = df.iloc[-1]
    near_ma = abs(last['close'] - df['ema21'].iloc[-1]) / last['close'] <= 0.02
    momentum_ok = last['rsi'] > 30
    return {'near_ma': near_ma, 'momentum_ok': momentum_ok, 'rsi': last['rsi']}

def execution_check_15m(df_15m):
    if detect_bullish_engulfing(df_15m): return {'signal': True, 'type': 'bullish_engulfing'}
    if detect_hammer(df_15m): return {'signal': True, 'type': 'hammer'}
    return {'signal': False}

def filter_symbols(tickers, base_quote='USDT', min_quote_vol=10000000, whitelist=None, blacklist=None):
    symbols = []
    for sym,data in tickers.items():
        if not sym.endswith(f'/{base_quote}'): continue
        if any(x in sym for x in ['UP/','DOWN/','BULL/','BEAR/']): continue
        if blacklist and sym in blacklist: continue
        if whitelist and sym not in whitelist: continue
        qvol = data.get('quoteVolume') or 0
        if qvol >= min_quote_vol:
            symbols.append((sym,qvol))
    symbols.sort(key=lambda x:x[1], reverse=True)
    return [s for s,_ in symbols]

class Broker:
    def __init__(self, exchange, paper=True):
        self.exchange = exchange
        self.paper = paper
        self.paper_positions = {}

    def fetch_quote_balance(self, quote='USDT'):
        bal = self.exchange.fetch_balance()
        total = bal['total'].get(quote, 0)
        free = bal['free'].get(quote, 0)
        return total, free

    def market_buy(self, symbol, amount_quote, markets):
        ticker = self.exchange.fetch_ticker(symbol)
        price = ticker['last']
        market = markets[symbol]
        step = market['limits']['amount']['step'] or 1
        qty = math.floor((amount_quote / price) / step) * step
        if qty <= 0: raise ValueError("Qty too small")
        if self.paper:
            order = {'id': f'paper-{symbol}-{int(time.time())}', 'symbol': symbol, 'price': price, 'amount': qty}
            self.paper_positions[symbol] = {'qty': qty, 'entry': price, 'time': datetime.utcnow().isoformat()}
            return order
        return self.exchange.create_order(symbol, 'market', 'buy', qty)

    def place_tp_sl(self, symbol, entry_price, qty, tp_pct, sl_pct, markets):
        tp = round(entry_price * (1 + tp_pct), 8)
        sl = round(entry_price * (1 - sl_pct), 8)
        if self.paper:
            self.paper_positions[symbol].update({'tp': tp, 'sl': sl})
            return {'tp': tp, 'sl': sl}
        self.exchange.create_order(symbol, 'limit', 'sell', qty, tp)
        params = {'stopPrice': sl, 'type': 'STOP_MARKET'}
        self.exchange.create_order(symbol, 'market', 'sell', qty, params=params)
        return {'tp': tp, 'sl': sl}

def main_loop():
    # Load environment from .env if present
    try:
        from dotenv import load_dotenv as _ld
        _ld()
    except Exception:
        pass
    cfg = {
        'BASE_QUOTE': os.getenv('BASE_QUOTE','USDT'),
        'UNIVERSE_SIZE': int(os.getenv('UNIVERSE_SIZE','12')),
        'MAX_OPEN_POSITIONS': int(os.getenv('MAX_OPEN_POSITIONS','5')),
        'POSITION_SIZING_MODE': os.getenv('POSITION_SIZING_MODE','risk'),
        'RISK_PER_TRADE': float(os.getenv('RISK_PER_TRADE','0.01')),
        'FIXED_ALLOCATION_USD': float(os.getenv('FIXED_ALLOCATION_USD','20')),
        'STOP_LOSS_PCT': float(os.getenv('STOP_LOSS_PCT','0.03')),
        'TAKE_PROFIT_PCT': float(os.getenv('TAKE_PROFIT_PCT','0.06')),
        'MIN_24H_QUOTE_VOLUME': float(os.getenv('MIN_24H_QUOTE_VOLUME','10000000')),
        'PAPER_TRADING': os.getenv('PAPER_TRADING','true').lower()=='true',
        'TESTNET': os.getenv('TESTNET','true').lower()=='true',
        'COOLDOWN_MINUTES': int(os.getenv('COOLDOWN_MINUTES','120')),
        'LOOP_INTERVAL_SECONDS': int(os.getenv('LOOP_INTERVAL_SECONDS','300')),
        'WHITELIST': os.getenv('WHITELIST','').split(',') if os.getenv('WHITELIST') else None,
        'BLACKLIST': set(os.getenv('BLACKLIST','').split(',')) if os.getenv('BLACKLIST') else None,
    }

    exchange = make_exchange(testnet=False)  # Switch to mainnet
    markets = exchange.load_markets()
    broker = Broker(exchange, paper=cfg['PAPER_TRADING'])

    cooldown = {}
    open_positions = 0

    while running:
        try:
            tickers = exchange.fetch_tickers()
            universe = filter_symbols(tickers, base_quote=cfg['BASE_QUOTE'], min_quote_vol=cfg['MIN_24H_QUOTE_VOLUME'], whitelist=cfg['WHITELIST'], blacklist=cfg['BLACKLIST'])[:cfg['UNIVERSE_SIZE']]
            print(f"[{datetime.utcnow().isoformat()}] Scanning universe: {universe}")
            total_quote, free_quote = broker.fetch_quote_balance(cfg['BASE_QUOTE'])
            equity = total_quote if total_quote>0 else 100.0
            for symbol in universe:
                if open_positions >= cfg['MAX_OPEN_POSITIONS']: break
                if symbol in cooldown and (datetime.utcnow() - cooldown[symbol]) < timedelta(minutes=cfg['COOLDOWN_MINUTES']): continue
                try:
                    df_daily = fetch_ohlcv_df(exchange, symbol, '1d', limit=300)
                    df_4h = fetch_ohlcv_df(exchange, symbol, '4h', limit=200)
                    df_15m = fetch_ohlcv_df(exchange, symbol, '15m', limit=200)
                except Exception as e:
                    print('fetch error', symbol, e); continue
                trend = daily_trend_check(df_daily)
                if not trend['bullish']:
                    print(symbol, 'skip: daily not bullish'); continue
                bias = bias_check_4h(df_4h)
                if not bias['near_ma'] or not bias['momentum_ok']:
                    print(symbol, 'skip: bias not OK'); continue
                exec_sig = execution_check_15m(df_15m)
                if not exec_sig['signal']:
                    print(symbol, 'no 15m confirmation'); continue
                last_price = float(df_15m['close'].iloc[-1])
                if cfg['POSITION_SIZING_MODE'] == 'fixed':
                    amount_quote = cfg['FIXED_ALLOCATION_USD']
                else:
                    risk_amount = equity * cfg['RISK_PER_TRADE']
                    stop_distance = last_price * cfg['STOP_LOSS_PCT']
                    if stop_distance <= 0:
                        print('invalid stop distance'); continue
                    qty = risk_amount / stop_distance
                    step = markets[symbol]['limits']['amount']['step'] or 1
                    qty = math.floor(qty / step) * step
                    if qty <= 0:
                        print('qty too small by risk sizing'); continue
                    amount_quote = qty * last_price
                try:
                    order = broker.market_buy(symbol, amount_quote=amount_quote, markets=markets)
                    entry = order.get('price') or last_price
                    amt = order.get('amount') if order.get('amount') else qty
                    levels = broker.place_tp_sl(symbol, entry, amt, cfg['TAKE_PROFIT_PCT'], cfg['STOP_LOSS_PCT'], markets)
                    log_trade_csv('logs/topdown_trades.csv', [datetime.utcnow().isoformat(), symbol, 'buy', amt, entry, levels['tp'], levels['sl'], f"topdown:{exec_sig['type']}", cfg['POSITION_SIZING_MODE']])
                    print('TRADE executed', symbol, entry, levels)
                    open_positions += 1
                    cooldown[symbol] = datetime.utcnow()
                except Exception as e:
                    print('order failed', e); continue
            # sleep
            total_sleep = cfg['LOOP_INTERVAL_SECONDS']
            slept = 0
            while slept < total_sleep and running:
                time.sleep(1); slept += 1
        except Exception as e:
            print('main loop error', e)
            time.sleep(5)
    print('Daemon exiting gracefully.')

if __name__ == '__main__':
    main_loop()
