# main_bot_final.py (CORREGIDO)

import time
import logging
import MetaTrader5 as mt5
import pandas as pd

# Importar nuestros módulos y configuraciones
import config as cfg
import mt5_manager as mt5_man
import state_manager as sm

# --- PARÁMETROS DE LA ESTRATEGIA V4 ---
ADX_THRESHOLD = 20
RSI_BUY_THRESHOLD = 50
RSI_SELL_THRESHOLD = 50

# --- Configuración de logging ---
logging.basicConfig(
    filename='bot_trading.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_trading_signal_v4(df):
    """
    Calcula la señal de trading basada en la estrategia V4 (Confluencia).
    El cálculo del ATR se ha movido fuera, al bucle principal.
    """
    if len(df) < cfg.ADX_PERIOD + 10:
        return "HOLD"
    
    # MACD
    ema_fast = df['close'].ewm(span=cfg.MACD_FAST, adjust=False).mean()
    ema_slow = df['close'].ewm(span=cfg.MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal_line = macd_line.ewm(span=cfg.MACD_SIGNAL, adjust=False).mean()
    macd_hist = macd_line - macd_signal_line
    
    # RSI
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=cfg.RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(span=cfg.RSI_PERIOD, adjust=False).mean()
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1] if avg_loss.iloc[-1] != 0 else float('inf')
    rsi = 100 - (100 / (1 + rs))

    # ADX
    # El cálculo del TR se hace fuera ahora, pero lo necesitamos para el ADX
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
    adx = dx.ewm(span=cfg.ADX_PERIOD, adjust=False).mean().iloc[-1]

    # Lógica de la señal V4
    signal = "HOLD"
    if adx > ADX_THRESHOLD:
        if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] < 0 and rsi > RSI_BUY_THRESHOLD:
            signal = "BUY"
        elif macd_hist.iloc[-1] < 0 and macd_hist.iloc[-2] > 0 and rsi < RSI_SELL_THRESHOLD:
            signal = "SELL"
    
    return signal

# --- BUCLE PRINCIPAL (main) ---
def main():
    print("🚀 Iniciando Bot de Trading V4 (Modo 'Forward Testing')...")
    if not mt5.initialize():
        logging.error(f"Error al inicializar MetaTrader5: {mt5.last_error()}")
        print(f"❌ ERROR: No se pudo inicializar MetaTrader5. Código: {mt5.last_error()}")
        return

    for sym in cfg.SYMBOLS:
        if not mt5.symbol_select(sym, True):
            logging.error(f"No se pudo seleccionar el símbolo {sym}. Error: {mt5.last_error()}")
            print(f"❌ ERROR: No se pudo seleccionar {sym}.")
            mt5.shutdown()
            return
    
    logging.info("Bot V4 iniciado. Monitoreando: %s", ", ".join(cfg.SYMBOLS))
    print(f"✅ Bot iniciado. Monitoreando: {', '.join(cfg.SYMBOLS)}")

    managed_trailing_stops = sm.load_trailing_stop_state()
    
    while True:
        try:
            account_info = mt5.account_info()
            if account_info is None:
                time.sleep(cfg.CHECK_INTERVAL)
                continue
            
            print(f"\n--- Nuevo ciclo --- Balance: {account_info.balance:.2f} {account_info.currency} ---")

            for symbol in cfg.SYMBOLS:
                print(f"\n--- Analizando {symbol} ---")
                
                positions = mt5.positions_get(symbol=symbol) or []
                my_positions = [p for p in positions if p.magic == cfg.MAGIC_NUMBER]

                # --- OBTENCIÓN Y PREPARACIÓN DE DATOS ---
                bars_needed = 100 
                df = mt5_man.get_rates(symbol, cfg.TIMEFRAME, bars_needed)
                if df.empty or len(df) < bars_needed:
                    print(f"[{symbol}] ⚠️ Datos insuficientes.")
                    continue
                
                # <-- CAMBIO CLAVE: Calcular ATR y añadirlo al DataFrame principal
                high_low = df['high'] - df['low']
                high_prev_close = abs(df['high'] - df['close'].shift())
                low_prev_close = abs(df['low'] - df['close'].shift())
                tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
                df['atr'] = tr.ewm(span=cfg.ATR_PERIOD, adjust=False).mean()
                atr_val = df['atr'].iloc[-1]
                # --- FIN DEL CAMBIO ---

                if not my_positions:
                    signal = get_trading_signal_v4(df)
                    print(f"[{symbol}] Señal generada: {signal}")
                    
                    if signal != "HOLD":
                        tick = mt5.symbol_info_tick(symbol)
                        if tick is None: continue
                        
                        lot = mt5_man.calculate_universal_lot_size(symbol, account_info, atr_val)
                        
                        if lot is not None and lot > 0:
                            if signal == "BUY":
                                sl = tick.ask - atr_val * cfg.SL_ATR_MULT
                                tp = tick.ask + atr_val * cfg.TP_ATR_MULT
                                mt5_man.open_position(symbol, mt5.ORDER_TYPE_BUY, lot, tick.ask, sl, tp)
                            elif signal == "SELL":
                                sl = tick.bid + atr_val * cfg.SL_ATR_MULT
                                tp = tick.bid - atr_val * cfg.TP_ATR_MULT
                                mt5_man.open_position(symbol, mt5.ORDER_TYPE_SELL, lot, tick.bid, sl, tp)
                
                positions_after = mt5.positions_get(symbol=symbol) or []
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    mt5_man.manage_trailing_stops(symbol, positions_after, atr_val, tick.ask, tick.bid, managed_trailing_stops)

        except Exception as e:
            logging.critical(f"Error crítico en el bucle principal: {e}", exc_info=True)
            print(f"🔥🔥🔥 ERROR CRÍTICO: {e}")
        finally:
            print(f"\n--- Ciclo finalizado. Esperando {cfg.CHECK_INTERVAL} segundos... ---")
            time.sleep(cfg.CHECK_INTERVAL)

if __name__ == "__main__":
    main()