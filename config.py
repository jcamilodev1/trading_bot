# config.py
import MetaTrader5 as mt5


ML_CONFIDENCE_THRESHOLD = 0.60 # Umbral de confianza (60%). Solo operamos si el modelo está más seguro que esto.
# --- Parámetros del bot (OPTIMIZADOS) ---------------------------------------
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]
TIMEFRAME = mt5.TIMEFRAME_M5

# === PARÁMETROS DE LA ESTRATEGIA V4 OPTIMIZADA ===
RSI_PERIOD = 19
MACD_FAST = 20
MACD_SLOW = 26
MACD_SIGNAL = 6
ADX_PERIOD = 19
# ===============================================

# Parámetros que no se optimizaron pero son parte de la V4
AO_FAST = 5
AO_SLOW = 34
BB_PERIOD = 20
BB_STD_DEV = 2
ATR_PERIOD = 14 # Usado para cálculo de lotaje

# --- Parámetros de Gestión de Riesgo (OPTIMIZADOS) ---
SL_ATR_MULT = 1.5441864660177191
TP_ATR_MULT = 2.504770076551997

# --- Configuración General del Bot ---
CHECK_INTERVAL = 60
MAGIC_NUMBER = 123456
FILLING_MODE = mt5.ORDER_FILLING_FOK
RISK_PERCENT = 0.005 # 0.5% de riesgo por operación. ¡MUY IMPORTANTE!
DEVIATION_PIPS = 20
STATE_FILE = 'trailing_stops_state.json'

# --- Parámetros de Trailing Stop ---
TRAILING_STOP_ACTIVE = True
TRAILING_STOP_DISTANCE_ATR = 1.0
MIN_PROFIT_TO_TRAIL_ATR = 0.5


ADX_THRESHOLD = 20
RSI_BUY_THRESHOLD = 50
RSI_SELL_THRESHOLD = 50
