#!/usr/bin/env python3
import ccxt, pandas as pd
from utils import ema, is_higher_highs_lows, detect_bullish_engulfing, detect_hammer
def fetch_ohlcv_df(exchange, symbol, timeframe, since=None, limit=1000):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df

def run_backtest(symbol, start, end):
    exchange = ccxt.binance({'enableRateLimit': True})
    df_daily = fetch_ohlcv_df(exchange, symbol, '1d', limit=1000)
    # Ensure both sides of comparison are timezone-naive
    df_daily['timestamp'] = df_daily['timestamp'].dt.tz_localize(None)
    df_daily = df_daily[(df_daily['timestamp']>=pd.to_datetime(start)) & (df_daily['timestamp']<=pd.to_datetime(end))].reset_index(drop=True)
    results = []
    for idx,row in df_daily.iterrows():
        day_start = row['timestamp']
        since_ms = int(day_start.timestamp()*1000)
        df_4h = fetch_ohlcv_df(exchange, symbol, '4h', since=since_ms, limit=1000)
        df_15m = fetch_ohlcv_df(exchange, symbol, '15m', since=since_ms, limit=1000)
        if df_4h.empty or df_15m.empty: continue
        daily_window = df_daily[max(0, idx-100):idx+1]
        if not (daily_window['close'].iloc[-1] > ema(daily_window['close'],21).iloc[-1] and is_higher_highs_lows(daily_window['close'])):
            continue
        bias = False
        df_4h['ema21'] = ema(df_4h['close'], 21)
        for i in range(len(df_4h)):
            last = df_4h.iloc[i]
            if abs(last['close'] - df_4h['ema21'].iloc[i]) / last['close'] <= 0.02:
                bias = True; break
        if not bias: continue
        for j in range(2, len(df_15m)):
            window = df_15m.iloc[:j+1]
            if detect_bullish_engulfing(window) or detect_hammer(window):
                entry = window.iloc[-1]['close']
                results.append({'day': day_start, 'entry': entry})
                break
    print(f"Backtest {symbol}: {len(results)} trades")
    return results

if __name__ == '__main__':
    run_backtest('SOL/USDT', '2022-01-01', '2023-12-31')
