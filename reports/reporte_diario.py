# reporte_diario.py
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta

# Importamos la configuración para usar el MAGIC_NUMBER
import config as cfg

def generar_reporte_diario():
    # --- Conexión a MT5 ---
    if not mt5.initialize():
        print("❌ initialize() falló, error code =", mt5.last_error())
        return

    # --- Definir el rango de fechas (el día de ayer completo) ---
    hoy = datetime.now()
    ayer = hoy - timedelta(days=1)
    fecha_inicio = datetime(ayer.year, ayer.month, ayer.day, 0, 0, 0)
    fecha_fin = datetime(ayer.year, ayer.month, ayer.day, 23, 59, 59)
    
    print(f"🚀 Generando reporte para el día: {fecha_inicio.strftime('%Y-%m-%d')}...")

    # --- Obtener el historial de tratos (deals) ---
    try:
        deals = mt5.history_deals_get(fecha_inicio, fecha_fin)
    except Exception as e:
        print(f"❌ Error al obtener el historial de tratos: {e}")
        mt5.shutdown()
        return

    if deals is None or len(deals) == 0:
        print("ℹ️ No se encontraron operaciones para el día especificado.")
        mt5.shutdown()
        return

    # --- Procesar los datos ---
    df_deals = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    
    # Filtrar solo por el número mágico de nuestro bot
    df_bot_deals = df_deals[df_deals['magic'] == cfg.MAGIC_NUMBER].copy()

    if df_bot_deals.empty:
        print(f"ℹ️ No se encontraron operaciones con el MAGIC_NUMBER {cfg.MAGIC_NUMBER}.")
        mt5.shutdown()
        return

    # Convertir el tiempo a un formato legible
    df_bot_deals['time'] = pd.to_datetime(df_bot_deals['time'], unit='s')
    
    # Seleccionar y renombrar columnas para mayor claridad
    columnas_reporte = {
        'time': 'Fecha y Hora',
        'symbol': 'Símbolo',
        'type': 'Tipo', # 0: BUY, 1: SELL
        'entry': 'Entrada/Salida', # 0: IN, 1: OUT, 2: IN/OUT
        'volume': 'Volumen',
        'price': 'Precio',
        'profit': 'Ganancia',
        'fee': 'Comisión',
        'swap': 'Swap',
        'order': 'Orden Ticket',
        'position_id': 'ID Posición'
    }
    df_reporte = df_bot_deals[columnas_reporte.keys()].rename(columns=columnas_reporte)

    # Reemplazar números por texto para mayor claridad
    df_reporte['Tipo'] = df_reporte['Tipo'].map({0: 'COMPRA', 1: 'VENTA'})
    df_reporte['Entrada/Salida'] = df_reporte['Entrada/Salida'].map({0: 'ENTRADA', 1: 'SALIDA', 2: 'ENTRADA/SALIDA'})

    # --- Guardar en Excel ---
    nombre_archivo = f"Reporte_Trades_{fecha_inicio.strftime('%Y-%m-%d')}.xlsx"
    try:
        df_reporte.to_excel(nombre_archivo, index=False, sheet_name='Trades')
        print(f"\n✅ ¡Éxito! Reporte guardado como '{nombre_archivo}'")
        
        # Calcular un resumen
        total_profit = df_reporte['Ganancia'].sum()
        total_trades = len(df_reporte[df_reporte['Entrada/Salida'] == 'SALIDA'])
        print("\n--- Resumen del Día ---")
        print(f"Operaciones cerradas: {total_trades}")
        print(f"Ganancia/Pérdida neta: {total_profit:.2f}")

    except Exception as e:
        print(f"❌ Error al guardar el archivo de Excel: {e}")

    # --- Desconexión ---
    mt5.shutdown()

if __name__ == "__main__":
    generar_reporte_diario()