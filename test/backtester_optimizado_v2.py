# backtester_optimizado_v4.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import MetaTrader5 as mt5
import numpy as np

import config as cfg

# --- PARÁMETROS ---
DATA_FILE_PATH = "EURUSD_5_data_1Y.csv" 
SYMBOL_FOR_INFO = "EURUSD"
ADX_THRESHOLD = 20
# Volvemos a los multiplicadores que mejor funcionaron
SL_MULT = 2.0
TP_MULT = 4.0

def run_optimized_backtest_v4():
    print(f"🚀 Iniciando Backtest V4 (Entrada por Confluencia MACD+RSI)...")

    # 1. Cargar y preparar datos (idéntico a V2)
    df = pd.read_csv(DATA_FILE_PATH, parse_dates=['time'])
    print(f"✅ Datos locales cargados: {len(df)} velas.")

    print("⏳ Pre-calculando indicadores...")
    # MACD
    ema_fast = df['close'].ewm(span=cfg.MACD_FAST, adjust=False).mean()
    ema_slow = df['close'].ewm(span=cfg.MACD_SLOW, adjust=False).mean()
    df['macd_line'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd_line'].ewm(span=cfg.MACD_SIGNAL, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']
    df['macd_hist_prev'] = df['macd_hist'].shift(1) # Necesitamos el valor anterior para detectar el cruce

    # RSI
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=cfg.RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(span=cfg.RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift())
    low_prev_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=cfg.ATR_PERIOD, adjust=False).mean()

    # ADX
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr_adx = pd.DataFrame({'tr': tr, 'plus_dm': plus_dm, 'minus_dm': abs(minus_dm)})
    atr_adx = tr_adx['tr'].ewm(span=cfg.ADX_PERIOD, adjust=False).mean()
    plus_di = 100 * (tr_adx['plus_dm'].ewm(span=cfg.ADX_PERIOD, adjust=False).mean() / atr_adx)
    minus_di = 100 * (tr_adx['minus_dm'].ewm(span=cfg.ADX_PERIOD, adjust=False).mean() / atr_adx)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    df['adx'] = dx.ewm(span=cfg.ADX_PERIOD, adjust=False).mean()
    
    df.dropna(inplace=True)
    print("✅ Indicadores calculados.")
    print("🏁 Iniciando simulación de trading...")
    
    trades = []
    open_trade = None

    for i in range(len(df)):
        current_row = df.iloc[i]
        
        if open_trade:
            # Lógica de cierre idéntica
            if (open_trade['type'] == 'BUY' and current_row['low'] <= open_trade['sl']) or \
               (open_trade['type'] == 'SELL' and current_row['high'] >= open_trade['sl']):
                open_trade.update({'exit_price': open_trade['sl'], 'exit_time': current_row['time']})
                trades.append(open_trade)
                open_trade = None
                continue

            if (open_trade['type'] == 'BUY' and current_row['high'] >= open_trade['tp']) or \
               (open_trade['type'] == 'SELL' and current_row['low'] <= open_trade['tp']):
                open_trade.update({'exit_price': open_trade['tp'], 'exit_time': current_row['time']})
                trades.append(open_trade)
                open_trade = None
                continue
        
        if not open_trade:
            # --- LÓGICA DE SEÑAL V4 POR CONFLUENCIA ---
            signal = "HOLD"
            
            # Condición de Compra:
            buy_condition = (
                current_row['adx'] > ADX_THRESHOLD and
                current_row['macd_hist'] > 0 and
                current_row['macd_hist_prev'] < 0 and # Detecta el cruce exacto
                current_row['rsi'] > 50 # <-- ¡CONFIRMACIÓN DE MOMENTUM!
            )
            
            # Condición de Venta:
            sell_condition = (
                current_row['adx'] > ADX_THRESHOLD and
                current_row['macd_hist'] < 0 and
                current_row['macd_hist_prev'] > 0 and # Detecta el cruce exacto
                current_row['rsi'] < 50 # <-- ¡CONFIRMACIÓN DE MOMENTUM!
            )

            if buy_condition:
                signal = "BUY"
            elif sell_condition:
                signal = "SELL"

            if signal != "HOLD":
                entry_price = current_row['open']
                atr_val = current_row['atr']
                
                if signal == "BUY":
                    sl = entry_price - atr_val * SL_MULT
                    tp = entry_price + atr_val * TP_MULT
                else: # SELL
                    sl = entry_price + atr_val * SL_MULT
                    tp = entry_price - atr_val * TP_MULT
                
                open_trade = {'type': signal, 'entry_price': entry_price, 'sl': sl, 'tp': tp, 'entry_time': current_row['time']}

    # Reporte de resultados
    if not mt5.initialize(): return
    symbol_info = mt5.symbol_info(SYMBOL_FOR_INFO)
    mt5.shutdown()
    if symbol_info is None: return

    print("\n--- 📊 Reporte de Backtesting (V4 - Confluencia MACD+RSI) ---")
    if not trades:
        print("No se realizó ninguna operación.")
        return

    df_trades = pd.DataFrame(trades)
    df_trades['profit'] = (df_trades['exit_price'] - df_trades['entry_price']) * np.where(df_trades['type'] == 'BUY', 1, -1)
    df_trades['pips'] = df_trades['profit'] / symbol_info.point
    
    total_trades = len(df_trades)
    winning_trades = df_trades[df_trades['profit'] > 0]
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    gross_profit = winning_trades['pips'].sum()
    gross_loss = abs(df_trades[df_trades['profit'] <= 0]['pips'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    print(f"Operaciones Totales:    {total_trades}")
    print(f"Tasa de Acierto:        {win_rate:.2f}%")
    print(f"Ganancia Neta (pips):   {df_trades['pips'].sum():.2f}")
    print(f"Profit Factor:          {profit_factor:.2f}  (>1.5 es bueno)")


if __name__ == "__main__":
    run_optimized_backtest_v4()