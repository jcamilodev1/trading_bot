# /indicators.py
import pandas as pd
from config import AO_FAST, AO_SLOW

# Nota: Hemos movido get_rates a mt5_manager porque interactúa directamente con MT5.

def get_rsi(df, period):
    if len(df) < period + 1: return None
    delta = df['close'].diff(1).dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    if avg_loss.iloc[-1] == 0:
        return 100.0 if avg_gain.iloc[-1] > 0 else 50.0
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
    return 100.0 - (100.0 / (1.0 + rs))

def get_vrsi(df, period):
    if len(df) < period + 1: return None
    delta = df['tick_volume'].diff(1).dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    if avg_loss.iloc[-1] == 0:
        return 100.0 if avg_gain.iloc[-1] > 0 else 50.0
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
    return 100.0 - (100.0 / (1.0 + rs))

def get_cci(df, period):
    if len(df) < period: return None
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: abs(x - x.mean()).mean(), raw=False)
    if md.iloc[-1] == 0: return 0
    return (tp.iloc[-1] - ma.iloc[-1]) / (0.015 * md.iloc[-1])

def get_vwap(df):
    if df['tick_volume'].sum() == 0: return None
    pv = df['close'] * df['tick_volume']
    return pv.cumsum().iloc[-1] / df['tick_volume'].cumsum().iloc[-1]

def get_ema(df, period):
    if len(df) < period: return None
    return df['close'].ewm(span=period, adjust=False).mean().iloc[-1]

def get_macd(df, fast, slow, signal):
    if len(df) < max(fast, slow, signal): return None
    fast_e = df['close'].ewm(span=fast, adjust=False).mean()
    slow_e = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = fast_e - slow_e
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    return (macd_line - sig_line).iloc[-1]

def get_bollinger(df, period, dev):
    if len(df) < period: return (None, None)
    sma = df['close'].rolling(period).mean().iloc[-1]
    std = df['close'].rolling(period).std().iloc[-1]
    if pd.isna(std): std = 0
    return sma + dev * std, sma - dev * std

def get_atr(df, period):
    if len(df) < period + 1: return None
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

def get_adx_di(df, period):
    if len(df) < period * 2: return (None, None, None)
    df_adx = df.copy()
    df_adx['tr1'] = df_adx['high'] - df_adx['low']
    df_adx['tr2'] = abs(df_adx['high'] - df_adx['close'].shift(1))
    df_adx['tr3'] = abs(df_adx['low'] - df_adx['close'].shift(1))
    df_adx['tr'] = df_adx[['tr1', 'tr2', 'tr3']].max(axis=1)
    df_adx['atr'] = df_adx['tr'].ewm(com=period - 1, adjust=False).mean()
    df_adx['plus_dm'] = df_adx['high'].diff()
    df_adx['minus_dm'] = df_adx['low'].diff()
    df_adx['plus_dm'] = df_adx.apply(lambda row: row['plus_dm'] if row['plus_dm'] > row['minus_dm'] and row['plus_dm'] > 0 else 0, axis=1)
    df_adx['minus_dm'] = df_adx.apply(lambda row: abs(row['minus_dm']) if row['minus_dm'] > row['plus_dm'] and row['minus_dm'] > 0 else 0, axis=1)
    df_adx['plus_dm_smooth'] = df_adx['plus_dm'].ewm(com=period - 1, adjust=False).mean()
    df_adx['minus_dm_smooth'] = df_adx['minus_dm'].ewm(com=period - 1, adjust=False).mean()
    epsilon = 1e-9
    df_adx['plus_di'] = 100 * (df_adx['plus_dm_smooth'] / (df_adx['atr'] + epsilon))
    df_adx['minus_di'] = 100 * (df_adx['minus_dm_smooth'] / (df_adx['atr'] + epsilon))
    df_adx['dx'] = 100 * (abs(df_adx['plus_di'] - df_adx['minus_di']) / (df_adx['plus_di'] + df_adx['minus_di'] + epsilon))
    df_adx['adx'] = df_adx['dx'].ewm(com=period - 1, adjust=False).mean()
    return df_adx['adx'].iloc[-1], df_adx['plus_di'].iloc[-1], df_adx['minus_di'].iloc[-1]

def get_ao(df):
    if len(df) < AO_SLOW: return None
    hl2 = (df['high'] + df['low']) / 2
    sma_fast = hl2.rolling(AO_FAST).mean()
    sma_slow = hl2.rolling(AO_SLOW).mean()
    return (sma_fast - sma_slow).iloc[-1]

def get_mfi(df, period=14):
    if len(df) < period + 1: return None
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['tick_volume']
    tp_prev = tp.shift(1)
    pos_mf = mf.where(tp > tp_prev, 0)
    neg_mf = mf.where(tp < tp_prev, 0)
    pos_mf_sum = pos_mf.rolling(period).sum().iloc[-1] if not pos_mf.empty else 0
    neg_mf_sum = neg_mf.rolling(period).sum().iloc[-1] if not neg_mf.empty else 0
    if neg_mf_sum == 0:
        return 100 if pos_mf_sum > 0 else 50
    rmf = pos_mf_sum / neg_mf_sum
    return 100 - (100 / (1 + rmf))