# ml_trainer_v3.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# --- PARÁMETROS ---
DATA_FILE_PATH = "EURUSD_M5_data_1Y.csv"
LOOK_AHEAD_PERIODS = 12
PROFIT_THRESHOLD_PIP = 15

def create_features_and_labels_v3(df):
    print("⏳ Creando un conjunto de features más rico (V3)...")
    
    # --- Parámetros de indicadores ---
    rsi_period = 19
    macd_fast = 20
    macd_slow = 26
    macd_signal = 6
    adx_period = 19
    atr_period = 14

    # --- Feature Engineering ---
    # RSI
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD completo
    ema_fast = df['close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=macd_slow, adjust=False).mean()
    df['macd_line'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd_line'].ewm(span=macd_signal, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    # ATR Normalizado (dividido por el precio de cierre para ser comparable)
    high_low = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift())
    low_prev_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df['atr_normalized'] = (tr.ewm(span=atr_period, adjust=False).mean()) / df['close']

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

    # --- Selección final de Features (X) ---
    feature_columns = ['rsi', 'macd_line', 'macd_signal', 'macd_hist', 'atr_normalized', 'adx']
    df.dropna(inplace=True)
    features = df[feature_columns]

    # --- Creación de Labels (y) ---
    print("⏳ Creando labels...")
    pip_value = 0.0001
    price_change = df['close'].shift(-LOOK_AHEAD_PERIODS) - df['close']
    conditions = [
        (price_change > PROFIT_THRESHOLD_PIP * pip_value),
        (price_change < -PROFIT_THRESHOLD_PIP * pip_value),
    ]
    choices = [1, -1]
    df['label'] = np.select(conditions, choices, default=0)
    labels = df['label']
    
    return features, labels

if __name__ == "__main__":
    print("Cargando datos...")
    df_full = pd.read_csv(DATA_FILE_PATH, parse_dates=['time'])
    
    X, y = create_features_and_labels_v3(df_full)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    print("\n⏳ Entrenando el modelo V3 (con más features)...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced', # Mantenemos el balanceo de clases
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("✅ Modelo entrenado.")

    print("\n--- 📊 Reporte de Evaluación del Modelo V3 ---")
    predictions = model.predict(X_test)
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, predictions))
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, predictions, target_names=['SELL (-1)', 'HOLD (0)', 'BUY (1)']))

    model_filename = 'trading_model_v3.joblib'
    joblib.dump(model, model_filename)
    print(f"\n✅ Modelo V3 guardado como '{model_filename}'")