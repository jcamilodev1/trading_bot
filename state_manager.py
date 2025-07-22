# /state_manager.py
import json
import logging
from config import STATE_FILE

def save_trailing_stop_state(state_dict):
    """Guarda el diccionario de estado del trailing stop en un archivo JSON."""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state_dict, f, indent=4)
    except Exception as e:
        logging.error(f"Error al guardar el estado del trailing stop: {e}")
        print(f"❌ Error al guardar estado: {e}")

def load_trailing_stop_state():
    """Carga el diccionario de estado del trailing stop desde un archivo JSON."""
    try:
        with open(STATE_FILE, 'r') as f:
            state_data = json.load(f)
            # Asegurarse de que las claves del diccionario sean enteros (ticket IDs)
            return {int(k): v for k, v in state_data.items()}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logging.error(f"Error al cargar el estado del trailing stop: {e}")
        print(f"❌ Error al cargar estado: {e}")
        return {}