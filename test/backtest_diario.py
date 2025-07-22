# backtest_diario.py (versión con detalles de trades)
import pandas as pd
import numpy as np
import joblib
import MetaTrader5 as mt5
from datetime import datetime

# --- PARÁMETROS (Sin cambios) ---
SYMBOL_TO_TEST = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
MODEL_FILE_PATH = "models/trading_filter_model.joblib"
ML_CONFIDENCE_THRESHOLD = 0.52 

ADX_THRESHOLD = 20
SL_MULT = 2.0
TP_MULT = 4.0
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14
ATR_PERIOD = 14

def run_daily_backtest():
    # ... (Toda la primera parte del script es idéntica: conexión, carga de datos, cálculo de indicadores y simulación) ...
    print(f"🚀 Iniciando Backtest para {SYMBOL_TO_TEST} en lo que va del día...")

    if not mt5.initialize():
        print("❌ initialize() falló.")
        return

    end_date = datetime.now()
    start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    try:
        ml_model = joblib.load(MODEL_FILE_PATH)
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró el archivo del modelo '{MODEL_FILE_PATH}'.")
        mt5.shutdown()
        return

    rates = mt5.copy_rates_range(SYMBOL_TO_TEST, TIMEFRAME, start_date, end_date)
    if rates is None or len(rates) == 0:
        print(f"ℹ️ No se encontraron datos para {SYMBOL_TO_TEST} en el día de hoy.")
        mt5.shutdown()
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"✅ Datos del día cargados: {len(df)} velas.")

    print("⏳ Pre-calculando indicadores...")
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

    print("🏁 Iniciando simulación de trading del día...")
    trades = []
    open_trade = None

    for i in range(len(df)):
        current_row = df.iloc[i]
        
        if open_trade:
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
            buy_candidate = (current_row['adx'] > ADX_THRESHOLD and current_row['macd_hist'] > 0 and
                             current_row['macd_hist_prev'] < 0 and current_row['rsi'] > 50)
            sell_candidate = (current_row['adx'] > ADX_THRESHOLD and current_row['macd_hist'] < 0 and
                              current_row['macd_hist_prev'] > 0 and current_row['rsi'] < 50)

            if buy_candidate or sell_candidate:
                features_columns = ['rsi', 'macd_hist', 'adx', 'atr_normalized']
                features_to_predict = pd.DataFrame([current_row[features_columns].values], columns=features_columns)
                probabilities = ml_model.predict_proba(features_to_predict)[0]
                confidence_in_winner = probabilities[1]

                if confidence_in_winner >= ML_CONFIDENCE_THRESHOLD:
                    signal = "BUY" if buy_candidate else "SELL"
                    entry_price = current_row['open']
                    atr_val = current_row['atr']
                    sl = entry_price - atr_val * SL_MULT if signal == "BUY" else entry_price + atr_val * SL_MULT
                    tp = entry_price + atr_val * TP_MULT if signal == "BUY" else entry_price - atr_val * TP_MULT
                    open_trade = {'type': signal, 'entry_price': entry_price, 'sl': sl, 'tp': tp, 'entry_time': current_row['time']}

    # --- SECCIÓN DE REPORTE MODIFICADA ---
    symbol_info = mt5.symbol_info(SYMBOL_TO_TEST)
    mt5.shutdown()

    print(f"\n--- 📊 Reporte de Backtesting del Día ({start_date.strftime('%Y-%m-%d')}) ---")
    if not trades:
        print("No se realizó ninguna operación en lo que va del día.")
        return

    df_trades = pd.DataFrame(trades)
    df_trades['profit'] = (df_trades['exit_price'] - df_trades['entry_price']) * np.where(df_trades['type'] == 'BUY', 1, -1)
    df_trades['pips'] = df_trades['profit'] / symbol_info.point
    
    # --- LÍNEAS AÑADIDAS PARA MOSTRAR DETALLES ---
    print("\n--- 📋 Detalles de las Operaciones ---")
    detalles_visibles = df_trades[['entry_time', 'type', 'entry_price', 'exit_price', 'pips']].copy()
    detalles_visibles['entry_time'] = detalles_visibles['entry_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    print(detalles_visibles.to_string(index=False))
    # --- FIN DE LÍNEAS AÑADIDAS ---

    print("\n--- Resumen General ---")
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
    run_daily_backtest()