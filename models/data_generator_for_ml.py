# data_generator_for_ml.py
import pandas as pd
import numpy as np

# --- PARÁMETROS DE LA ESTRATEGIA V4 ---
# Usamos la configuración de la V4 original, no la optimizada, para tener más datos.
ADX_THRESHOLD = 20
SL_MULT = 2.0
TP_MULT = 4.0
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14
ATR_PERIOD = 14
DATA_FILE_PATH = "EURUSD_M5_data_1Y.csv"
OUTPUT_DATA_FILE = "v4_trades_for_ml.csv"

def generate_trade_data():
    print(f"🚀 Generando datos de trades de la estrategia V4...")
    df = pd.read_csv(DATA_FILE_PATH, parse_dates=['time'])
    
    # --- Pre-cálculo de Indicadores ---
    # ... (Copiamos exactamente los mismos cálculos de indicadores del backtester V4)
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
    
    # --- Simulación para capturar los datos de cada trade ---
    trades_data = []
    open_trade = None

    for i in range(len(df)):
        current_row = df.iloc[i]
        
        # Lógica de cierre para determinar si el trade anterior ganó o perdió
        if open_trade:
            is_winner = False
            if (open_trade['type'] == 'BUY' and current_row['high'] >= open_trade['tp']) or \
               (open_trade['type'] == 'SELL' and current_row['low'] <= open_trade['tp']):
                is_winner = True
            
            # Si el trade se cerró (por SL o TP), lo guardamos
            if (is_winner or (open_trade['type'] == 'BUY' and current_row['low'] <= open_trade['sl']) or \
               (open_trade['type'] == 'SELL' and current_row['high'] >= open_trade['sl'])):
                
                # Guardamos los features del momento de la ENTRADA
                trade_features = open_trade['entry_features']
                # Añadimos el resultado del trade como LABEL
                trade_features['is_winner'] = 1 if is_winner else 0
                trades_data.append(trade_features)
                open_trade = None
        
        # Lógica de apertura V4
        if not open_trade:
            buy_condition = (current_row['adx'] > ADX_THRESHOLD and current_row['macd_hist'] > 0 and
                             current_row['macd_hist_prev'] < 0 and current_row['rsi'] > 50)
            sell_condition = (current_row['adx'] > ADX_THRESHOLD and current_row['macd_hist'] < 0 and
                              current_row['macd_hist_prev'] > 0 and current_row['rsi'] < 50)

            if buy_condition or sell_condition:
                signal = "BUY" if buy_condition else "SELL"
                entry_price = current_row['open']
                atr_val = current_row['atr']
                sl = entry_price - atr_val * SL_MULT if signal == "BUY" else entry_price + atr_val * SL_MULT
                tp = entry_price + atr_val * TP_MULT if signal == "BUY" else entry_price - atr_val * TP_MULT
                
                # Guardamos los features de este momento para el futuro
                entry_features = {
                    'rsi': current_row['rsi'],
                    'macd_hist': current_row['macd_hist'],
                    'adx': current_row['adx'],
                    'atr_normalized': current_row['atr'] / current_row['close']
                }
                open_trade = {'type': signal, 'tp': tp, 'sl': sl, 'entry_features': entry_features}

    # Guardar los datos en un CSV
    df_ml = pd.DataFrame(trades_data)
    df_ml.to_csv(OUTPUT_DATA_FILE, index=False)
    
    print(f"\n✅ ¡Éxito! Se generó el archivo '{OUTPUT_DATA_FILE}' con {len(df_ml)} trades.")
    print("Distribución de resultados:")
    print(df_ml['is_winner'].value_counts())

if __name__ == "__main__":
    generate_trade_data()