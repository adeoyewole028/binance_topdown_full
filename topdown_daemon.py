#!/usr/bin/env python3
import os, signal, time, math
from datetime import datetime
import pandas as pd
from runtime_environment import RuntimeEnvironment, construct_exchange, load_launcher_environment, parse_boolean
from utils import ema, rsi, is_higher_highs_lows, detect_bullish_engulfing, detect_hammer
from strategy_profiles import evaluate_observation as evaluate_market_snapshot

__all__ = ['evaluate_market_snapshot']

running = True
def signal_handler(sig, frame):
    global running
    print(f"Received signal {sig}, shutting down...")
    running = False

def make_exchange(testnet=None):
    runtime = RuntimeEnvironment.from_mapping(os.environ)
    if testnet is not None and testnet != (runtime.venue == 'testnet'):
        raise ValueError('Explicit venue contradicts configured venue')
    return construct_exchange(runtime, private=runtime.mode == 'exchange')

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
    load_launcher_environment()
    if not parse_boolean(os.getenv('AUTOMATION_ENABLED', 'false')):
        raise ValueError('Automation is disabled by default')
    # BT03 retires the separate CSV/in-memory writer; BT09 preserves this
    # workflow by attaching strategy evaluation to the shared owned intake.
    raise ValueError('Automation requires the shared execution owner integration in BT09')

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main_loop()
