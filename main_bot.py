# /main_bot.py
import time
import logging
import MetaTrader5 as mt5

# Importar nuestros módulos y configuraciones
import config as cfg
import indicators as ind
import mt5_manager as mt5_man
import signal_generator as sig
import state_manager as sm

# --- Configuración de logging -----------------------------------------------
logging.basicConfig(
    filename='bot_trading.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def check_exit_conditions(position, ind_data):
    """
    Verifica si hay condiciones para cerrar una posición abierta basadas en indicadores.
    """
    is_buy = position.type == mt5.POSITION_TYPE_BUY
    ema_short = ind_data.get('ema20')
    current_bid = ind_data.get('bid_price')
    current_ask = ind_data.get('ask_price')

    if ema_short and current_bid and current_ask:
        if is_buy and current_bid < ema_short:
            print(f"[{position.symbol}] 📉 Señal de salida (Compra): Precio (BID) cruzó por debajo de EMA20.")
            logging.info(f"[{position.symbol}] Salida dinámica (BUY): Precio {current_bid} < EMA20 {ema_short}")
            return True
        if not is_buy and current_ask > ema_short:
            print(f"[{position.symbol}] 📈 Señal de salida (Venta): Precio (ASK) cruzó por encima de EMA20.")
            logging.info(f"[{position.symbol}] Salida dinámica (SELL): Precio {current_ask} > EMA20 {ema_short}")
            return True

    macd_hist = ind_data.get('macd')
    if macd_hist is not None:
        if is_buy and macd_hist < 0:
            print(f"[{position.symbol}] 📉 Señal de salida (Compra): MACD cruzó a negativo.")
            logging.info(f"[{position.symbol}] Salida dinámica (BUY): MACD {macd_hist} < 0")
            return True
        if not is_buy and macd_hist > 0:
            print(f"[{position.symbol}] 📈 Señal de salida (Venta): MACD cruzó a positivo.")
            logging.info(f"[{position.symbol}] Salida dinámica (SELL): MACD {macd_hist} > 0")
            return True
            
    return False

def main():
    print("🚀 Iniciando Bot de Trading v3.0 (Modular)...")
    if not mt5.initialize():
        logging.error(f"Error al inicializar MetaTrader5: {mt5.last_error()}")
        print(f"❌ ERROR: No se pudo inicializar MetaTrader5. Código: {mt5.last_error()}")
        return

    for sym in cfg.SYMBOLS:
        if not mt5.symbol_select(sym, True):
            logging.error(f"No se pudo seleccionar el símbolo {sym}. Error: {mt5.last_error()}")
            print(f"❌ ERROR: No se pudo seleccionar {sym}. Verifique si existe en Market Watch.")
            mt5.shutdown()
            return

    logging.info("Bot de trading modular iniciado. Monitoreando símbolos: %s", ", ".join(cfg.SYMBOLS))
    print(f"✅ Bot iniciado. Monitoreando: {', '.join(cfg.SYMBOLS)}")

    managed_trailing_stops = sm.load_trailing_stop_state()
    print(f"ℹ️ {len(managed_trailing_stops)} posiciones cargadas desde el estado de Trailing Stop.")

    while True:
        try:
            account_info = mt5.account_info()
            if account_info is None:
                logging.error(f"No se pudo obtener la info de la cuenta: {mt5.last_error()}")
                print(f"❌ ERROR: No se pudo obtener la info de la cuenta. Reintentando...")
                time.sleep(cfg.CHECK_INTERVAL)
                continue
            
            print(f"\n--- Nuevo ciclo de análisis --- Balance: {account_info.balance:.2f} {account_info.currency} ---")

            for symbol in cfg.SYMBOLS:
                print(f"\n--- Analizando {symbol} ---")
                
                bars_needed = max(cfg.EMA_LONG_PERIOD, cfg.BB_PERIOD, cfg.ATR_PERIOD, cfg.RSI_PERIOD, cfg.MACD_SLOW, cfg.ADX_PERIOD, cfg.AO_SLOW) + 5
                df = mt5_man.get_rates(symbol, cfg.TIMEFRAME, bars_needed)

                if df.empty or len(df) < bars_needed - 1:
                    print(f"[{symbol}] ⚠️ Datos insuficientes ({len(df)} barras).")
                    continue

                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    print(f"[{symbol}] ⚠️ No se pudo obtener el precio (tick) actual.")
                    continue
                
                ask, bid = tick.ask, tick.bid
                print(f"[{symbol}] Precio actual: ASK={ask:.5f}, BID={bid:.5f}")
                
                atr_val = ind.get_atr(df, cfg.ATR_PERIOD)
                
                indicators_data = {
                    'ask_price': ask, 'bid_price': bid,
                    'rsi': ind.get_rsi(df, cfg.RSI_PERIOD), 'vrsi': ind.get_vrsi(df, cfg.VRSI_PERIOD),
                    'cci': ind.get_cci(df, cfg.CCI_PERIOD), 'ema20': ind.get_ema(df, cfg.EMA_SHORT_PERIOD),
                    'ema50': ind.get_ema(df, cfg.EMA_LONG_PERIOD),
                    'macd': ind.get_macd(df, cfg.MACD_FAST, cfg.MACD_SLOW, cfg.MACD_SIGNAL),
                    'bbu': ind.get_bollinger(df, cfg.BB_PERIOD, cfg.BB_STD_DEV)[0], 
                    'bbl': ind.get_bollinger(df, cfg.BB_PERIOD, cfg.BB_STD_DEV)[1],
                    'atr': atr_val, 'adx': ind.get_adx_di(df, cfg.ADX_PERIOD)[0],
                    'di+': ind.get_adx_di(df, cfg.ADX_PERIOD)[1], 'di-': ind.get_adx_di(df, cfg.ADX_PERIOD)[2],
                    'ao': ind.get_ao(df), 'mfi': ind.get_mfi(df, cfg.RSI_PERIOD), 'vwap': ind.get_vwap(df),
                }

                positions = mt5.positions_get(symbol=symbol) or []
                my_positions = [p for p in positions if p.magic == cfg.MAGIC_NUMBER]

                if not my_positions:
                    signal = sig.get_gpt_signal({k: v for k, v in indicators_data.items() if v is not None})
                    
                    if signal in ("BUY", "SELL"):
                        lot = mt5_man.calculate_universal_lot_size(symbol, account_info, atr_val)
                        if lot is not None and lot > 0 and atr_val is not None and atr_val > 0:
                            print(f"[{symbol}] Cálculo de lote - Lote final: {lot:.2f}")
                            if signal == "BUY":
                                sl = ask - atr_val * cfg.SL_ATR_MULT
                                tp = ask + atr_val * cfg.TP_ATR_MULT
                                print(f"[{symbol}] ➡️ Abriendo LONG. P:{ask:.5f}, SL:{sl:.5f}, TP:{tp:.5f}")
                                success, result = mt5_man.open_position(symbol, mt5.ORDER_TYPE_BUY, lot, ask, sl, tp)
                                if success and cfg.TRAILING_STOP_ACTIVE and result:
                                    managed_trailing_stops[result.order] = sl
                                    sm.save_trailing_stop_state(managed_trailing_stops)
                            elif signal == "SELL":
                                sl = bid + atr_val * cfg.SL_ATR_MULT
                                tp = bid - atr_val * cfg.TP_ATR_MULT
                                print(f"[{symbol}] ⬅️ Abriendo SHORT. P:{bid:.5f}, SL:{sl:.5f}, TP:{tp:.5f}")
                                success, result = mt5_man.open_position(symbol, mt5.ORDER_TYPE_SELL, lot, bid, sl, tp)
                                if success and cfg.TRAILING_STOP_ACTIVE and result:
                                    managed_trailing_stops[result.order] = sl
                                    sm.save_trailing_stop_state(managed_trailing_stops)
                        else:
                             print(f"[{symbol}] ⚠️ No se pudo calcular el tamaño del lote o el ATR es inválido.")
                else:
                    print(f"[{symbol}] ℹ️ Ya hay posiciones abiertas. Chequeando condiciones de salida y Trailing Stop.")
                    for p in my_positions:
                        if check_exit_conditions(p, indicators_data):
                            print(f"[{symbol}] 🟢 Cerrando posición {p.ticket} por condiciones de salida dinámicas.")
                            price_type = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                            mt5_man.close_position(p, symbol, price_type)

                positions_after_actions = mt5.positions_get(symbol=symbol) or []
                mt5_man.manage_trailing_stops(symbol, positions_after_actions, atr_val, ask, bid, managed_trailing_stops)

        except Exception as e:
            logging.critical(f"Error crítico en el bucle principal: {e}", exc_info=True)
            print(f"🔥🔥🔥 ERROR CRÍTICO EN BUCLE PRINCIPAL: {e}")
        finally:
            print(f"\n--- Ciclo finalizado. Esperando {cfg.CHECK_INTERVAL} segundos... ---")
            time.sleep(cfg.CHECK_INTERVAL)

    mt5.shutdown()
    logging.info("Bot de trading finalizado.")
    print("👋 Bot de trading finalizado.")

if __name__ == "__main__":
    main()