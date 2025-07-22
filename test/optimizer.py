import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# optimizer.py
import pandas as pd
import numpy as np
import optuna
import MetaTrader5 as mt5

# --- CONFIGURACIÓN ---
DATA_FILE_PATH = "EURUSD_5_data_1Y.csv"
SYMBOL_FOR_INFO = "EURUSD"
N_TRIALS = 100 # Número de combinaciones a probar. Empieza con 50-100.

def objective(trial):
    """
    Esta es la función que Optuna intentará maximizar.
    Cada 'trial' es una ejecución del backtest con una nueva combinación de parámetros.
    """
    # 1. Definimos los parámetros que Optuna va a probar y sus rangos
    adx_period = trial.suggest_int('adx_period', 10, 20)
    adx_threshold = trial.suggest_int('adx_threshold', 18, 30)
    rsi_period = trial.suggest_int('rsi_period', 10, 20)

    macd_fast = trial.suggest_int('macd_fast', 8, 20)
    macd_slow = trial.suggest_int('macd_slow', 21, 35)
    macd_signal = trial.suggest_int('macd_signal', 6, 12)

    sl_mult = trial.suggest_float('sl_mult', 1.5, 3.0)
    tp_mult = trial.suggest_float('tp_mult', 2.5, 6.0)

    # 2. Cargamos los datos
    df = pd.read_csv(DATA_FILE_PATH, parse_dates=['time'])

    # 3. Calculamos los indicadores con los parámetros del 'trial' actual
    # MACD
    ema_fast = df['close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=macd_slow, adjust=False).mean()
    df['macd_line'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd_line'].ewm(span=macd_signal, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']
    df['macd_hist_prev'] = df['macd_hist'].shift(1)

    # RSI
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR (usamos un periodo fijo para no complicar demasiado)
    high_low = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift())
    low_prev_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=14, adjust=False).mean()

    # ADX
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr_adx = pd.DataFrame({'tr': tr, 'plus_dm': plus_dm, 'minus_dm': abs(minus_dm)})
    atr_adx = tr_adx['tr'].ewm(span=adx_period, adjust=False).mean()
    plus_di = 100 * (tr_adx['plus_dm'].ewm(span=adx_period, adjust=False).mean() / atr_adx)
    minus_di = 100 * (tr_adx['minus_dm'].ewm(span=adx_period, adjust=False).mean() / atr_adx)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    df['adx'] = dx.ewm(span=adx_period, adjust=False).mean()

    df.dropna(inplace=True)

    # 4. Ejecutamos la simulación (lógica de la V4)
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
            buy_condition = (current_row['adx'] > adx_threshold and current_row['macd_hist'] > 0 and
                             current_row['macd_hist_prev'] < 0 and current_row['rsi'] > 50)
            sell_condition = (current_row['adx'] > adx_threshold and current_row['macd_hist'] < 0 and
                              current_row['macd_hist_prev'] > 0 and current_row['rsi'] < 50)

            if buy_condition or sell_condition:
                signal = "BUY" if buy_condition else "SELL"
                entry_price = current_row['open']
                atr_val = current_row['atr']
                sl = entry_price - atr_val * sl_mult if signal == "BUY" else entry_price + atr_val * sl_mult
                tp = entry_price + atr_val * tp_mult if signal == "BUY" else entry_price - atr_val * tp_mult
                open_trade = {'type': signal, 'entry_price': entry_price, 'sl': sl, 'tp': tp, 'entry_time': current_row['time']}

    # 5. Calculamos y devolvemos el resultado a optimizar (Profit Factor)
    if not trades:
        return 0.0 # Si no hay trades, el PF es 0

    df_trades = pd.DataFrame(trades)
    df_trades['profit'] = (df_trades['exit_price'] - df_trades['entry_price']) * np.where(df_trades['type'] == 'BUY', 1, -1)

    gross_profit = df_trades[df_trades['profit'] > 0]['profit'].sum()
    gross_loss = abs(df_trades[df_trades['profit'] <= 0]['profit'].sum())

    if gross_loss == 0:
        return float('inf') # Evitar división por cero

    profit_factor = gross_profit / gross_loss
    return profit_factor

if __name__ == "__main__":
    # Creamos el "estudio" de optimización
    # Le decimos que queremos maximizar el resultado de la función 'objective'
    study = optuna.create_study(direction="maximize")

    # Lanzamos la optimización
    study.optimize(objective, n_trials=N_TRIALS)

    # Imprimimos los resultados
    print("\n" + "="*50)
    print("OPTIMIZACIÓN FINALIZADA")
    print(f"Mejor Profit Factor encontrado: {study.best_value:.4f}")
    print("Mejores Parámetros:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*50)