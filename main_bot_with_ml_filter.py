# main_bot_con_logs.py
import pandas as pd
import MetaTrader5 as mt5
import time
import logging
import joblib

# Importar nuestros módulos y configuraciones
import config as cfg
import mt5_manager as mt5_man
import state_manager as sm

# --- Cargar el modelo de filtro entrenado ---
# (Esta parte no cambia)
try:
    ml_model = joblib.load('trading_filter_model.joblib')
    print("✅ Modelo de Machine Learning cargado exitosamente.")
except FileNotFoundError:
    print("❌ ERROR: No se encontró el archivo del modelo 'trading_filter_model.joblib'.")
    ml_model = None

# --- Lógica de la Estrategia V4 (para generar candidatos) ---
# (Esta función no cambia)
def get_v4_signal_candidate(df):
    # ... (código idéntico al anterior)
    # MACD
    ema_fast = df['close'].ewm(span=cfg.MACD_FAST, adjust=False).mean()
    ema_slow = df['close'].ewm(span=cfg.MACD_SLOW, adjust=False).mean()
    df['macd_hist'] = (ema_fast - ema_slow) - (ema_fast - ema_slow).ewm(span=cfg.MACD_SIGNAL, adjust=False).mean()
    df['macd_hist_prev'] = df['macd_hist'].shift(1)
    # RSI
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=cfg.RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(span=cfg.RSI_PERIOD, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + avg_gain / avg_loss))
    # ADX
    high_low = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift())
    low_prev_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
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
    df['atr_normalized'] = (tr.ewm(span=cfg.ATR_PERIOD, adjust=False).mean()) / df['close']

    df.dropna(inplace=True)
    if df.empty:
        return "HOLD", None
        
    current_row = df.iloc[-1]

    # Lógica de la señal V4
    buy_candidate = (current_row['adx'] > 20 and current_row['macd_hist'] > 0 and
                     current_row['macd_hist_prev'] < 0 and current_row['rsi'] > 50)
    sell_candidate = (current_row['adx'] > 20 and current_row['macd_hist'] < 0 and
                      current_row['macd_hist_prev'] > 0 and current_row['rsi'] < 50)
    
    if buy_candidate:
        return "BUY", df.iloc[[-1]] 
    if sell_candidate:
        return "SELL", df.iloc[[-1]]
        
    return "HOLD", None


# --- BUCLE PRINCIPAL CON LOGS DETALLADOS ---
def main():
    if ml_model is None:
        return
        
    print("🚀 Iniciando Bot Híbrido Final (Reglas V4 + Filtro ML)...")
    if not mt5.initialize():
        print("❌ ERROR: No se pudo inicializar MetaTrader5.")
        return

    # ... (código de inicialización de MT5 idéntico)
    for symbol in cfg.SYMBOLS:
        if not mt5.symbol_select(symbol, True):
            print(f"❌ ERROR: No se pudo seleccionar {symbol}.")
            mt5.shutdown()
            return
            
    print(f"✅ Bot iniciado. Monitoreando: {', '.join(cfg.SYMBOLS)}")

    while True:
        try:
            account_info = mt5.account_info()
            if account_info is None:
                print("⚠️ No se pudo obtener la info de la cuenta. Reintentando...")
                time.sleep(cfg.CHECK_INTERVAL)
                continue

            print(f"\n--- Nuevo ciclo --- Balance: {account_info.balance:.2f} {account_info.currency} ---")

            for symbol in cfg.SYMBOLS:
                print(f"\n--- Analizando {symbol} ---")
                
                positions = mt5.positions_get(symbol=symbol) or []
                my_positions = [p for p in positions if p.magic == cfg.MAGIC_NUMBER]

                if not my_positions:
                    print(f"[{symbol}] ℹ️ No hay posiciones abiertas. Buscando nueva señal...") # <-- LOG ADICIONAL
                    
                    bars_needed = 100
                    print(f"[{symbol}] 📈 Obteniendo {bars_needed} velas para análisis...") # <-- LOG ADICIONAL
                    df = mt5_man.get_rates(symbol, cfg.TIMEFRAME, bars_needed)
                    
                    if df.empty or len(df) < bars_needed:
                        print(f"[{symbol}] ⚠️ Datos insuficientes para el análisis. Saltando.") # <-- LOG ADICIONAL
                        continue

                    # 1. Obtener señal candidata de la estrategia V4
                    signal_candidate, features_df = get_v4_signal_candidate(df)
                    print(f"[{symbol}] 🤖 Resultado del análisis de reglas V4: {signal_candidate}") # <-- LOG ADICIONAL
                    
                    if signal_candidate != "HOLD":
                        print(f"[{symbol}] 🔍 Señal candidata detectada. Pasando al filtro de ML...") # <-- LOG ADICIONAL
                        
                        features = features_df[['rsi', 'macd_hist', 'adx', 'atr_normalized']]
                        
                        probabilities = ml_model.predict_proba(features)[0]
                        confidence_in_winner = probabilities[1] 
                        
                        print(f"[{symbol}] 🧠 Confianza del modelo ML en el éxito: {confidence_in_winner:.2%}")
                        
                        if confidence_in_winner > cfg.ML_CONFIDENCE_THRESHOLD:
                            print(f"[{symbol}] ✅ Confianza suficiente. Ejecutando operación.")
                            tick = mt5.symbol_info_tick(symbol)
                            if tick is None: continue

                            atr_val = features_df['atr_normalized'].iloc[-1] * tick.ask
                            lot = mt5_man.calculate_universal_lot_size(symbol, account_info, atr_val)
                            
                            if signal_candidate == "BUY":
                                sl = tick.ask - atr_val * cfg.SL_ATR_MULT
                                tp = tick.ask + atr_val * cfg.TP_ATR_MULT
                                mt5_man.open_position(symbol, mt5.ORDER_TYPE_BUY, lot, tick.ask, sl, tp)
                            else: # SELL
                                sl = tick.bid + atr_val * cfg.SL_ATR_MULT
                                tp = tick.bid - atr_val * cfg.TP_ATR_MULT
                                mt5_man.open_position(symbol, mt5.ORDER_TYPE_SELL, lot, tick.bid, sl, tp)
                        else:
                            print(f"[{symbol}] ❌ Confianza insuficiente. Operación filtrada por el modelo ML.")
                else:
                    print(f"[{symbol}] ℹ️ Posición abierta detectada. Saltando búsqueda de señal.") # <-- LOG ADICIONAL

                # Gestionar Trailing Stops
                print(f"[{symbol}] 🔒 Gestionando Trailing Stop para posiciones existentes...") # <-- LOG ADICIONAL
                positions_after = mt5.positions_get(symbol=symbol) or []
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    # Necesitamos el ATR para el trailing stop, lo recalculamos si es necesario
                    if 'df' not in locals() or df.empty:
                        df = mt5_man.get_rates(symbol, cfg.TIMEFRAME, 20)
                        if not df.empty:
                             high_low = df['high'] - df['low']
                             high_prev_close = abs(df['high'] - df['close'].shift())
                             low_prev_close = abs(df['low'] - df['close'].shift())
                             tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
                             current_atr = tr.ewm(span=cfg.ATR_PERIOD, adjust=False).mean().iloc[-1]
                             mt5_man.manage_trailing_stops(symbol, positions_after, current_atr, tick.ask, tick.bid, sm.load_trailing_stop_state())
                    else:
                        current_atr = (df['atr_normalized'].iloc[-1] * df['close'].iloc[-1])
                        mt5_man.manage_trailing_stops(symbol, positions_after, current_atr, tick.ask, tick.bid, sm.load_trailing_stop_state())


        except Exception as e:
            logging.critical(f"Error crítico en el bucle principal: {e}", exc_info=True)
            print(f"🔥🔥🔥 ERROR CRÍTICO: {e}")
        finally:
            print(f"\n--- Ciclo finalizado. Esperando {cfg.CHECK_INTERVAL} segundos... ---")
            time.sleep(cfg.CHECK_INTERVAL)

if __name__ == "__main__":
    main()