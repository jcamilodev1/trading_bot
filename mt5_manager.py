# /mt5_manager.py
import MetaTrader5 as mt5
import pandas as pd
import logging
import time
from config import *
from state_manager import save_trailing_stop_state

def get_rates(symbol, timeframe, bars):
    """Obtiene datos de velas de MT5 y los convierte a un DataFrame de Pandas."""
    try:
        data = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if data is None or len(data) == 0:
            logging.warning(f"[{symbol}] No se pudieron obtener datos históricos. Error: {mt5.last_error()}")
            print(f"[{symbol}] ⚠️ Advertencia: No se pudieron obtener datos históricos.")
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    except Exception as e:
        logging.error(f"[{symbol}] Error al obtener o procesar datos de velas: {e}")
        print(f"[{symbol}] ❌ Error: Fallo al procesar datos de velas.")
        return pd.DataFrame()

def send_trade_request(request, symbol):
    for i in range(MAX_RETRIES):
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            action_desc = request.get('action_description', request['action'])
            logging.info(f"[{symbol}] Operación exitosa. Tipo: {action_desc}, Volumen: {request['volume']:.2f}. Ticket: {result.order}. Retcode: {result.retcode}")
            print(f"✅ [{symbol}] Operación exitosa. Ticket: {result.order}, Tipo: {action_desc}, Vol: {request['volume']:.2f}")
            return True, result
        else:
            logging.error(f"[{symbol}] Falló la operación ({i+1}/{MAX_RETRIES}). Error: {result.retcode} - {result.comment}. Solicitud: {request}")
            print(f"❌ [{symbol}] Falló la operación. Código: {result.retcode}, Comentario: {result.comment}. Reintento {i+1}/{MAX_RETRIES}")
            if result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                logging.warning(f"[{symbol}] Requote detectado. Reintentando...")
                time.sleep(1)
            elif result.retcode == mt5.TRADE_RETCODE_NO_CHANGES:
                logging.info(f"[{symbol}] No se requiere ningún cambio o la orden ya fue procesada: {result.comment}")
                return True, result
            else:
                time.sleep(1)
    logging.error(f"[{symbol}] Fallo definitivo de la operación después de {MAX_RETRIES} reintentos.")
    print(f"🔴 [{symbol}] Fallo definitivo después de {MAX_RETRIES} intentos.")
    return False, None

def calculate_universal_lot_size(symbol, account_info, atr_val):
    try:
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logging.error(f"[{symbol}] No se pudo obtener symbol_info para el cálculo de lote.")
            return None

        stop_loss_distance = atr_val * SL_ATR_MULT
        contract_size = symbol_info.trade_contract_size
        quote_currency = symbol_info.currency_margin
        loss_in_quote_currency = stop_loss_distance * contract_size
        account_currency = account_info.currency
        loss_in_account_currency = loss_in_quote_currency

        if quote_currency != account_currency:
            conversion_pair_forward = f"{quote_currency}{account_currency}"
            conversion_pair_backward = f"{account_currency}{quote_currency}"
            tick_forward = mt5.symbol_info_tick(conversion_pair_forward)
            if tick_forward and tick_forward.ask > 0:
                loss_in_account_currency = loss_in_quote_currency * tick_forward.ask
            else:
                tick_backward = mt5.symbol_info_tick(conversion_pair_backward)
                if tick_backward and tick_backward.bid > 0:
                    loss_in_account_currency = loss_in_quote_currency / tick_backward.bid
                else:
                    logging.warning(f"[{symbol}] No se encontró par de conversión ({conversion_pair_forward} o {conversion_pair_backward}) para el cálculo de lote.")
                    return None

        if loss_in_account_currency <= 0:
            logging.warning(f"[{symbol}] El riesgo calculado por lote es cero o negativo.")
            return symbol_info.volume_min

        risk_amount = account_info.balance * RISK_PERCENT
        desired_lot = risk_amount / loss_in_account_currency
        lot = max(symbol_info.volume_min, min(desired_lot, symbol_info.volume_max))
        step = symbol_info.volume_step
        if step > 0:
            lot = round(lot / step) * step
        lot = max(symbol_info.volume_min, lot)
        return lot
    except Exception as e:
        logging.error(f"[{symbol}] Excepción en calculate_universal_lot_size: {e}")
        return None

def open_position(symbol, trade_type, lot, price, sl, tp):
    action_description = "ABRIR COMPRA" if trade_type == mt5.ORDER_TYPE_BUY else "ABRIR VENTA"
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lot, "type": trade_type,
        "price": price, "sl": sl, "tp": tp, "deviation": DEVIATION_PIPS, "magic": MAGIC_NUMBER,
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": FILLING_MODE,
        "action_description": action_description
    }
    return send_trade_request(request, symbol)

def close_position(position, symbol, price_type):
    current_tick = mt5.symbol_info_tick(symbol)
    if current_tick is None:
        logging.error(f"[{symbol}] No se pudo obtener el tick para cerrar posición.")
        print(f"❌ [{symbol}] No se pudo obtener el precio actual para cerrar la posición {position.ticket}.")
        return False, None

    price_to_use = current_tick.ask if price_type == mt5.ORDER_TYPE_BUY else current_tick.bid
    action_description = "CERRAR VENTA" if price_type == mt5.ORDER_TYPE_BUY else "CERRAR COMPRA"
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": position.volume,
        "type": price_type, "position": position.ticket, "price": price_to_use,
        "deviation": DEVIATION_PIPS, "magic": MAGIC_NUMBER, "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": FILLING_MODE, "action_description": action_description
    }
    return send_trade_request(request, symbol)


def manage_trailing_stops(symbol, current_positions, atr_val, current_ask, current_bid, managed_trailing_stops_dict):
    if not TRAILING_STOP_ACTIVE or atr_val is None or atr_val <= 0:
        return

    trailing_distance_points = atr_val * TRAILING_STOP_DISTANCE_ATR
    min_profit_for_trail_points = atr_val * MIN_PROFIT_TO_TRAIL_ATR
    active_tickets_for_symbol = {pos.ticket for pos in current_positions if pos.symbol == symbol}
    state_changed = False

    for pos in current_positions:
        if pos.symbol != symbol or pos.magic != MAGIC_NUMBER:
            continue
        
        if pos.ticket not in managed_trailing_stops_dict:
            managed_trailing_stops_dict[pos.ticket] = pos.sl
            state_changed = True
            logging.info(f"[{symbol}] Posición {pos.ticket} añadida a gestión de TS. SL actual: {pos.sl:.5f}")
            print(f"[{symbol}] ℹ️ Posición {pos.ticket} añadida a gestión de Trailing Stop.")

        last_known_sl = managed_trailing_stops_dict.get(pos.ticket, pos.sl)
        is_buy = pos.type == mt5.POSITION_TYPE_BUY
        
        if is_buy:
            profit_condition = (current_bid - pos.price_open) >= min_profit_for_trail_points
            new_sl_potential = current_bid - trailing_distance_points
            sl_improvement_condition = new_sl_potential > last_known_sl
            valid_sl_price = new_sl_potential < current_bid
        else:
            profit_condition = (pos.price_open - current_ask) >= min_profit_for_trail_points
            new_sl_potential = current_ask + trailing_distance_points
            sl_improvement_condition = new_sl_potential < last_known_sl
            valid_sl_price = new_sl_potential > current_ask

        if profit_condition and sl_improvement_condition and valid_sl_price:
            request = {
                "action": mt5.TRADE_ACTION_SLTP, "symbol": symbol, "position": pos.ticket,
                "sl": new_sl_potential, "tp": pos.tp, "magic": MAGIC_NUMBER,
                "action_description": f"Actualizar Trailing SL ({'Compra' if is_buy else 'Venta'})"
            }
            success, _ = send_trade_request(request, symbol)
            if success:
                managed_trailing_stops_dict[pos.ticket] = new_sl_potential
                state_changed = True
                logging.info(f"[{symbol}] Trailing SL de {'Compra' if is_buy else 'Venta'} actualizado para {pos.ticket} a {new_sl_potential:.5f}")
                print(f"[{symbol}] {'📈' if is_buy else '📉'} Trailing SL actualizado para {'Compra' if is_buy else 'Venta'} {pos.ticket} a {new_sl_potential:.5f}")

    tickets_to_remove = [t for t in managed_trailing_stops_dict if t not in active_tickets_for_symbol]
    if tickets_to_remove:
        for t in tickets_to_remove:
            del managed_trailing_stops_dict[t]
            logging.info(f"[{symbol}] Posición {t} eliminada de la gestión de Trailing Stop (cerrada).")
        state_changed = True

    if state_changed:
        save_trailing_stop_state(managed_trailing_stops_dict)