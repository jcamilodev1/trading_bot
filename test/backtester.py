# backtester_local.py


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import MetaTrader5 as mt5

# Importamos nuestros módulos y configuraciones
import config as cfg
import indicators as ind

# --- PARÁMETROS DEL BACKTEST ---
# Apunta al archivo CSV que generaste en el paso anterior
DATA_FILE_PATH = "EURUSD_5_data_1Y.csv" 
SYMBOL_FOR_INFO = "EURUSD" # Símbolo para obtener info de pips
EMA_TREND_PERIOD = 200     # Usaremos una EMA larga para definir la tendencia principal

def get_technical_signal(df):
    """
    Genera una señal de trading basada en reglas técnicas, sin IA.
    Usa una copia del dataframe para evitar SettingWithCopyWarning.
    """
    df_copy = df.copy()
    
    # 1. Filtro de Tendencia
    df_copy['ema_trend'] = df_copy['close'].ewm(span=EMA_TREND_PERIOD, adjust=False).mean()
    last_close = df_copy['close'].iloc[-1]
    last_ema_trend = df_copy['ema_trend'].iloc[-1]
    
    # 2. Señal de MACD
    macd_hist = ind.get_macd(df_copy, cfg.MACD_FAST, cfg.MACD_SLOW, cfg.MACD_SIGNAL)
    
    # 3. Filtro RSI
    rsi = ind.get_rsi(df_copy, cfg.RSI_PERIOD)

    if macd_hist is None or rsi is None:
        return "HOLD"
        
    is_uptrend = last_close > last_ema_trend
    is_downtrend = last_close < last_ema_trend

    if is_uptrend and macd_hist > 0 and rsi < 70:
        return "BUY"
    
    if is_downtrend and macd_hist < 0 and rsi > 30:
        return "SELL"
        
    return "HOLD"

def run_local_backtest():
    """Función principal que ejecuta el backtest desde un archivo CSV local."""
    print(f"🚀 Iniciando Backtest desde archivo local: {DATA_FILE_PATH}...")

    # 1. Cargar datos desde el archivo CSV
    try:
        df_history = pd.read_csv(DATA_FILE_PATH)
        # MUY IMPORTANTE: Convertir la columna de tiempo a formato datetime
        df_history['time'] = pd.to_datetime(df_history['time'])
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró el archivo de datos '{DATA_FILE_PATH}'.")
        print("Asegúrate de que el archivo está en la misma carpeta y el nombre es correcto.")
        return

    # Breve conexión a MT5 solo para obtener datos del símbolo (valor del pip)
    if not mt5.initialize():
        print("❌ No se pudo conectar a MT5 para obtener la info del símbolo.")
        symbol_info = None
    else:
        symbol_info = mt5.symbol_info(SYMBOL_FOR_INFO)
        mt5.shutdown() # Nos desconectamos inmediatamente

    if symbol_info is None:
        print(f"❌ No se pudo obtener la info para {SYMBOL_FOR_INFO}. El cálculo de pips puede fallar.")
        return

    print(f"✅ Datos locales cargados: {len(df_history)} velas.")
    
    # 2. Simulación de trading (esta lógica es idéntica a la anterior)
    trades = []
    open_trade = None
    bars_needed = max(EMA_TREND_PERIOD, cfg.MACD_SLOW) + 50

    for i in range(bars_needed, len(df_history)):
        current_data = df_history.iloc[:i]
        current_price_open = df_history['open'][i]
        current_price_high = df_history['high'][i]
        current_price_low = df_history['low'][i]

        if open_trade:
            # Chequeo de SL/TP
            if (open_trade['type'] == 'BUY' and current_price_low <= open_trade['sl']) or \
               (open_trade['type'] == 'SELL' and current_price_high >= open_trade['sl']):
                open_trade['exit_price'] = open_trade['sl']
                open_trade['exit_time'] = df_history['time'][i]
                open_trade['profit'] = (open_trade['exit_price'] - open_trade['entry_price']) * (1 if open_trade['type'] == 'BUY' else -1)
                trades.append(open_trade)
                open_trade = None
                continue

            if (open_trade['type'] == 'BUY' and current_price_high >= open_trade['tp']) or \
               (open_trade['type'] == 'SELL' and current_price_low <= open_trade['tp']):
                open_trade['exit_price'] = open_trade['tp']
                open_trade['exit_time'] = df_history['time'][i]
                open_trade['profit'] = (open_trade['exit_price'] - open_trade['entry_price']) * (1 if open_trade['type'] == 'BUY' else -1)
                trades.append(open_trade)
                open_trade = None
                continue

        if not open_trade:
            signal = get_technical_signal(current_data)
            
            if signal != "HOLD":
                atr_val = ind.get_atr(current_data, cfg.ATR_PERIOD)
                if atr_val is None or atr_val <= 0:
                    continue

                if signal == "BUY":
                    entry_price = current_price_open
                    sl = entry_price - atr_val * cfg.SL_ATR_MULT
                    tp = entry_price + atr_val * cfg.TP_ATR_MULT
                    open_trade = {'type': 'BUY', 'entry_price': entry_price, 'sl': sl, 'tp': tp, 'entry_time': df_history['time'][i]}
                
                elif signal == "SELL":
                    entry_price = current_price_open
                    sl = entry_price + atr_val * cfg.SL_ATR_MULT
                    tp = entry_price - atr_val * cfg.TP_ATR_MULT
                    open_trade = {'type': 'SELL', 'entry_price': entry_price, 'sl': sl, 'tp': tp, 'entry_time': df_history['time'][i]}

    # 3. Analizar y mostrar resultados
    print("\n--- 📊 Reporte de Backtesting (Local) ---")
    if not trades:
        print("No se realizó ninguna operación.")
        return

    df_trades = pd.DataFrame(trades)
    df_trades['pips'] = df_trades['profit'] / symbol_info.point
    
    total_trades = len(df_trades)
    winning_trades = df_trades[df_trades['profit'] > 0]
    losing_trades = df_trades[df_trades['profit'] <= 0]
    
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    gross_profit = winning_trades['pips'].sum()
    gross_loss = abs(losing_trades['pips'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    net_profit_pips = df_trades['pips'].sum()

    print(f"Operaciones Totales:    {total_trades}")
    print(f"Operaciones Ganadoras:  {len(winning_trades)}")
    print(f"Operaciones Perdedoras: {len(losing_trades)}")
    print(f"Tasa de Acierto:        {win_rate:.2f}%")
    print(f"Ganancia Neta (pips):   {net_profit_pips:.2f}")
    print(f"Profit Factor:          {profit_factor:.2f}  (>1.5 es bueno)")

if __name__ == "__main__":
    run_local_backtest()