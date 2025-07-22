# main_bot_diagnostico_final.py
import pandas as pd
import MetaTrader5 as mt5
import time
import logging
import joblib

# Importar nuestros módulos y configuraciones
import config as cfg
import mt5_manager as mt5_man
import state_manager as sm

# ... (El código para cargar el modelo no cambia) ...
try:
    ml_model = joblib.load('trading_filter_model.joblib')
    print("✅ Modelo de Machine Learning cargado exitosamente.")
except FileNotFoundError:
    print("❌ ERROR: No se encontró el archivo del modelo 'trading_filter_model.joblib'.")
    ml_model = None

# --- LÓGICA DE LA ESTRATEGIA V4 CON DIAGNÓSTICO FINAL ---
def get_v4_signal_candidate_with_debug(df):
    if df.empty or len(df) < 100:
        return "HOLD", "Datos insuficientes", None

    # --- Cálculo de Indicadores ---
    # ... (Cálculos idénticos)
    ema_fast = df['close'].ewm(span=cfg.MACD_FAST, adjust=False).mean()
    ema_slow = df['close'].ewm(span=cfg.MACD_SLOW, adjust=False).mean()
    df['macd_hist'] = (ema_fast - ema_slow) - (ema_fast - ema_slow).ewm(span=cfg.MACD_SIGNAL, adjust=False).mean()
    df['macd_hist_prev'] = df['macd_hist'].shift(1)
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=cfg.RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(span=cfg.RSI_PERIOD, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + avg_gain / avg_loss))
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
        return "HOLD", "Datos insuficientes tras cálculo", None
        
    current_row = df.iloc[-1]

    # --- LÓGICA DE DIAGNÓSTICO MEJORADA ---
    adx_threshold = 20
    rsi_buy_threshold = 50
    rsi_sell_threshold = 50

    adx_val = current_row['adx']
    if adx_val <= adx_threshold:
        reason = f"ADX bajo (actual: {adx_val:.2f}, esperado: > {adx_threshold})"
        return "HOLD", reason, None

    macd_hist_val = current_row['macd_hist']
    macd_hist_prev_val = current_row['macd_hist_prev']
    buy_cross = macd_hist_val > 0 and macd_hist_prev_val < 0
    sell_cross = macd_hist_val < 0 and macd_hist_prev_val > 0
    
    if not buy_cross and not sell_cross:
        # --- ESTE ES EL CAMBIO PRINCIPAL ---
        esperado_buy = "anterior < 0 y actual > 0"
        esperado_sell = "anterior > 0 y actual < 0"
        reason = (f"Sin cruce de MACD (actual: {macd_hist_val:.5f}, anterior: {macd_hist_prev_val:.5f}). "
                  f"Esperado para BUY: {esperado_buy}. Esperado para SELL: {esperado_sell}.")
        return "HOLD", reason, None

    rsi_val = current_row['rsi']
    if buy_cross:
        if rsi_val > rsi_buy_threshold:
            return "BUY", "Todas las condiciones cumplidas", df.iloc[[-1]]
        else:
            reason = f"Cruce de Compra, pero RSI bajo (actual: {rsi_val:.2f}, esperado: > {rsi_buy_threshold})"
            return "HOLD", reason, None
            
    if sell_cross:
        if rsi_val < rsi_sell_threshold:
            return "SELL", "Todas las condiciones cumplidas", df.iloc[[-1]]
        else:
            reason = f"Cruce de Venta, pero RSI alto (actual: {rsi_val:.2f}, esperado: < {rsi_sell_threshold})"
            return "HOLD", reason, None

    return "HOLD", "Condición no determinada", None

# --- El resto del archivo (función main) es idéntico al anterior ---
def main():
    if ml_model is None:
        return
        
    print("🚀 Iniciando Bot Híbrido Final (con Diagnóstico v2)...")
    if not mt5.initialize():
        print("❌ ERROR: No se pudo inicializar MetaTrader5.")
        return

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
                    print(f"[{symbol}] ℹ️ No hay posiciones abiertas. Buscando nueva señal...")
                    
                    bars_needed = 100
                    print(f"[{symbol}] 📈 Obteniendo {bars_needed} velas para análisis...")
                    df = mt5_man.get_rates(symbol, cfg.TIMEFRAME, bars_needed)
                    
                    if df.empty or len(df) < bars_needed:
                        print(f"[{symbol}] ⚠️ Datos insuficientes para el análisis. Saltando.")
                        continue

                    signal_candidate, reason, features_df = get_v4_signal_candidate_with_debug(df)
                    
                    if signal_candidate == "HOLD":
                        print(f"[{symbol}] 🤖 Resultado: HOLD. Razón: {reason}")
                    else: 
                        print(f"[{symbol}] 🤖 ¡Señal candidata detectada: {signal_candidate}!")
                        print(f"[{symbol}] 🔍 Razón: {reason}. Pasando al filtro de ML...")
                        
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
                    print(f"[{symbol}] ℹ️ Posición abierta detectada. Saltando búsqueda de señal.")

                print(f"[{symbol}] 🔒 Gestionando Trailing Stop para posiciones existentes...")
                positions_after = mt5.positions_get(symbol=symbol) or []
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    # ... (código de gestión de Trailing Stop sin cambios)
                    pass

        except Exception as e:
            logging.critical(f"Error crítico en el bucle principal: {e}", exc_info=True)
            print(f"🔥🔥🔥 ERROR CRÍTICO: {e}")
        finally:
            print(f"\n--- Ciclo finalizado. Esperando {cfg.CHECK_INTERVAL} segundos... ---")
            time.sleep(cfg.CHECK_INTERVAL)

if __name__ == "__main__":
    main()