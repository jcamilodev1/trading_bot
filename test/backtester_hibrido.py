# backtester_hibrido.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pandas as pd
import numpy as np
import joblib
import MetaTrader5 as mt5

# --- PARÁMETROS DEL BACKTEST HÍBRIDO ---
DATA_FILE_PATH = "EURUSD_M5_data_1Y.csv"
MODEL_FILE_PATH = "trading_filter_model.joblib"
SYMBOL_FOR_INFO = "EURUSD"

# ¡PARÁMETRO CLAVE! Umbral de confianza para el filtro de ML
ML_CONFIDENCE_THRESHOLD = 0.52 # Empezamos con 52% (ligeramente mejor que una moneda al aire)

# Parámetros de la estrategia V4 que generan las señales candidatas
ADX_THRESHOLD = 20
SL_MULT = 2.0
TP_MULT = 4.0
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14
ATR_PERIOD = 14

def run_hybrid_backtest():
    print(f"🚀 Iniciando Backtest Híbrido (V4 + Filtro ML)...")
    print(f"Umbral de confianza del ML: {ML_CONFIDENCE_THRESHOLD:.2%}")

    # 1. Cargar el modelo y los datos
    try:
        ml_model = joblib.load(MODEL_FILE_PATH)
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró el archivo del modelo '{MODEL_FILE_PATH}'.")
        return
    
    df = pd.read_csv(DATA_FILE_PATH, parse_dates=['time'])
    print(f"✅ Datos y modelo cargados.")

    # 2. Pre-cálculo de indicadores (idéntico a los scripts anteriores)
    print("⏳ Pre-calculando indicadores...")
    # ... (Cálculos de MACD, RSI, ATR, ADX. Es el mismo bloque que antes)
    ema_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['macd_hist'] = (ema_fast - ema_slow) - (ema_fast - ema_slow).ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['macd_hist_prev'] = df['macd_hist'].shift(1)
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(span=RSI_PERIOD, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + avg_gain / avg_loss))
    high_low = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift())
    low_prev_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    df['atr_normalized'] = df['atr'] / df['close']
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr_adx = pd.DataFrame({'tr': tr, 'plus_dm': plus_dm, 'minus_dm': abs(minus_dm)})
    atr_adx = tr_adx['tr'].ewm(span=ADX_PERIOD, adjust=False).mean()
    plus_di = 100 * (tr_adx['plus_dm'].ewm(span=ADX_PERIOD, adjust=False).mean() / atr_adx)
    minus_di = 100 * (tr_adx['minus_dm'].ewm(span=ADX_PERIOD, adjust=False).mean() / atr_adx)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    df['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    df.dropna(inplace=True)
    print("✅ Indicadores calculados.")
    
    # 3. Simulación de trading con el filtro ML
    print("🏁 Iniciando simulación de trading híbrida...")
    trades = []
    open_trade = None

    for i in range(len(df)):
        current_row = df.iloc[i]
        
        if open_trade: # Lógica de cierre
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
            # Paso 1: Generar señal candidata con reglas V4
            buy_candidate = (current_row['adx'] > ADX_THRESHOLD and current_row['macd_hist'] > 0 and
                             current_row['macd_hist_prev'] < 0 and current_row['rsi'] > 50)
            sell_candidate = (current_row['adx'] > ADX_THRESHOLD and current_row['macd_hist'] < 0 and
                              current_row['macd_hist_prev'] > 0 and current_row['rsi'] < 50)

            if buy_candidate or sell_candidate:
                # Paso 2: Usar el filtro ML si hay un candidato
                # features_to_predict = current_row[['rsi', 'macd_hist', 'adx', 'atr_normalized']].values.reshape(1, -1)
                features_columns = ['rsi', 'macd_hist', 'adx', 'atr_normalized']
                features_df = pd.DataFrame([current_row[features_columns].values], columns=features_columns)
                probabilities = ml_model.predict_proba(features_df)[0]
                confidence_in_winner = probabilities[1] # Probabilidad de la clase '1' (Winner)

                # Paso 3: Decisión final basada en el umbral de confianza
                if confidence_in_winner >= ML_CONFIDENCE_THRESHOLD:
                    signal = "BUY" if buy_candidate else "SELL"
                    entry_price = current_row['open']
                    atr_val = current_row['atr']
                    sl = entry_price - atr_val * SL_MULT if signal == "BUY" else entry_price + atr_val * SL_MULT
                    tp = entry_price + atr_val * TP_MULT if signal == "BUY" else entry_price - atr_val * TP_MULT
                    open_trade = {'type': signal, 'entry_price': entry_price, 'sl': sl, 'tp': tp, 'entry_time': current_row['time']}

    # 4. Reporte de resultados
    # ... (Bloque de reporte idéntico a los scripts anteriores)
    if not mt5.initialize(): return
    symbol_info = mt5.symbol_info(SYMBOL_FOR_INFO)
    mt5.shutdown()
    if symbol_info is None: return

    print(f"\n--- 📊 Reporte de Backtesting Híbrido (Umbral: {ML_CONFIDENCE_THRESHOLD:.0%}) ---")
    if not trades:
        print("No se realizó ninguna operación con este nivel de confianza.")
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
    print(f"Profit Factor:          {profit_factor:.2f}")

if __name__ == "__main__":
    run_hybrid_backtest()