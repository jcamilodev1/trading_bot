# download_historical.py

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta # Librería para manejar fechas fácilmente

# --- PARÁMETROS ---
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
YEARS_TO_DOWNLOAD = 1 # Años de datos que quieres descargar hacia atrás desde hoy

# --- FUNCIÓN PRINCIPAL DE DESCARGA ---
def download_data_in_chunks():
    """
    Descarga datos históricos en chunks para evitar los límites de MT5 y los une.
    """
    print(f"Iniciando descarga de {YEARS_TO_DOWNLOAD} año(s) de datos para {SYMBOL}...")

    # 1. Conectar a MetaTrader 5
    if not mt5.initialize():
        print("initialize() falló, error code =", mt5.last_error())
        return

    # 2. Definir el rango de fechas total
    end_date = datetime.now()
    start_date = end_date - relativedelta(years=YEARS_TO_DOWNLOAD)
    print(f"Rango de fechas objetivo: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")

    all_data_chunks = []
    current_date = end_date

    # 3. Bucle para descargar los datos en chunks mensuales hacia atrás
    while current_date > start_date:
        # Definimos el inicio del chunk (un mes antes)
        chunk_start = current_date - relativedelta(months=1)
        # Nos aseguramos de no pasarnos de la fecha de inicio total
        if chunk_start < start_date:
            chunk_start = start_date

        print(f"  -> Descargando chunk: {chunk_start.strftime('%Y-%m-%d')} a {current_date.strftime('%Y-%m-%d')}")

        # Pedimos los datos para este chunk
        rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, chunk_start, current_date)

        if rates is not None and len(rates) > 0:
            all_data_chunks.append(pd.DataFrame(rates))
        else:
            print(f"  -> No se recibieron datos para este chunk.")
        
        # Movemos la fecha actual al inicio del chunk que acabamos de descargar
        current_date = chunk_start

    # 4. Unir, limpiar y guardar los datos
    if not all_data_chunks:
        print("❌ No se pudo descargar ningún dato en el rango especificado.")
        mt5.shutdown()
        return

    # Unimos todos los dataframes de la lista en uno solo
    final_df = pd.concat(all_data_chunks, ignore_index=True)

    # Limpieza de datos
    final_df['time'] = pd.to_datetime(final_df['time'], unit='s')
    final_df.sort_values('time', inplace=True)
    # MUY IMPORTANTE: Eliminar duplicados que puedan ocurrir en los bordes de los chunks
    final_df.drop_duplicates(subset='time', keep='first', inplace=True)

    # 5. Guardar en CSV
    output_filename = f'{SYMBOL}_{TIMEFRAME}_data_{YEARS_TO_DOWNLOAD}Y.csv'
    final_df.to_csv(output_filename, index=False)

    print("\n----------------------------------------------------")
    print(f"✅ ¡Éxito! Se guardaron {len(final_df)} velas en '{output_filename}'")
    print(f"Rango final de datos: de {final_df['time'].iloc[0]} a {final_df['time'].iloc[-1]}")
    print("----------------------------------------------------")

    mt5.shutdown()

# --- Ejecutar el script ---
if __name__ == "__main__":
    download_data_in_chunks()