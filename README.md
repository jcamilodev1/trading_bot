# Bot Machine

Este proyecto es un bot de trading automatizado para MetaTrader 5, diseñado para operar en los pares de divisas EURUSD, GBPUSD y USDJPY utilizando múltiples indicadores técnicos y gestión de riesgos avanzada.

## Características principales
- Integración con MetaTrader 5 (MT5)
- Soporte para varios indicadores técnicos: RSI, CCI, EMA, MACD, Bandas de Bollinger, ATR, ADX, Awesome Oscillator, Keltner Channel
- Gestión de riesgo configurable por operación
- Soporte para trailing stop
- Backtesting y optimización de estrategias
- Configuración flexible mediante el archivo `config.py`

## Estructura del proyecto
- `main_bot_with_ml_filter.py`: Script principal del bot de trading con filtro de Machine Learning
- `main_bot.py` / `main_bot_final.py`: Scripts alternativos del bot de trading
- `config.py`: Configuración de parámetros del bot e indicadores
- `indicators.py`: Implementación de indicadores técnicos
- `mt5_manager.py`: Funciones para interactuar con MetaTrader 5
- `signal_generator.py`: Generación de señales de trading
- `state_manager.py`: Gestión del estado y trailing stops
- `requirements.txt`: Dependencias del proyecto
- `test/`: Scripts de pruebas y backtesting

## Requisitos
- Python 3.12+
- MetaTrader5 (paquete Python)
- Tener MetaTrader 5 instalado y configurado

Instala las dependencias con:
```powershell
pip install -r requirements.txt
```

## Ejecución
Ejecuta el bot principal con filtro ML:
```powershell
python main_bot_with_ml_filter.py
```

## Backtesting
Ejecuta los scripts de backtesting desde la carpeta `test` para probar estrategias con datos históricos.

## Configuración
Modifica los parámetros en `config.py` para ajustar indicadores, gestión de riesgo y comportamiento del bot.

## Licencia
Este proyecto se distribuye bajo la licencia MIT.
