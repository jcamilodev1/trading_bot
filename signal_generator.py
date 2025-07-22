# /signal_generator.py
import os
import logging
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_gpt_signal(ind_data):
    prompt = f"Analiza los siguientes datos de indicadores técnicos para el mercado actual. " \
             f"Tu objetivo es identificar una oportunidad de trading clara. " \
             f"Considera la tendencia, el impulso, la volatilidad y los niveles clave de soporte/resistencia. " \
             f"Si la dirección es fuertemente alcista y la entrada es favorable, genera 'BUY'. " \
             f"Si la dirección es fuertemente bajista y la entrada es favorable, genera 'SELL'. " \
             f"Si no hay una señal clara, el mercado está consolidado, o hay alta incertidumbre, genera 'HOLD'. " \
             f"Responde ÚNICAMENTE con una de las tres palabras: BUY, SELL, o HOLD.\n\n" \
             f"Datos: {', '.join(f'{k}={v:.5f}' for k, v in ind_data.items() if v is not None)}"
    try:
        logging.info(f"Enviando a GPT: {prompt}")
        print(f"🌍 Enviando datos a GPT para análisis...")
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un analista de trading cuantitativo y experto en el mercado. Tu análisis es puramente técnico, basado en los indicadores proporcionados. Tu respuesta debe ser una señal de trading directa y sin ambigüedades."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=5
        )
        sig = resp.choices[0].message.content.strip().upper().split()[0]
        if sig in ("BUY", "SELL", "HOLD"):
            print(f"✨ Señal de GPT recibida: {sig}")
            return sig
        else:
            logging.warning(f"Respuesta inesperada de GPT: '{sig}'. Usando 'HOLD'.")
            print(f"⚠️ GPT dio una respuesta inesperada: '{sig}'. Usando HOLD por seguridad.")
            return "HOLD"
    except openai.APIError as e:
        logging.error(f"Error de la API de OpenAI: {e}. Usando 'HOLD'.")
        print(f"❌ Error de la API de OpenAI: {e}. Usando HOLD.")
        return "HOLD"
    except Exception as e:
        logging.error(f"Error inesperado al obtener señal de GPT: {e}. Usando 'HOLD'.")
        print(f"❌ Error inesperado al obtener señal de GPT: {e}. Usando HOLD.")
        return "HOLD"