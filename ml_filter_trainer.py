# ml_filter_trainer.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# --- PARÁMETROS ---
DATA_FILE = "v4_trades_for_ml.csv"

if __name__ == "__main__":
    # 1. Cargar el dataset de trades
    print(f"Cargando datos de trades desde '{DATA_FILE}'...")
    df = pd.read_csv(DATA_FILE)

    # 2. Definir Features (X) y Labels (y)
    # Las features son los indicadores que guardamos
    features = ['rsi', 'macd_hist', 'adx', 'atr_normalized']
    # La label es si el trade fue ganador o no
    label = 'is_winner'

    X = df[features]
    y = df[label]

    # 3. Dividir los datos para entrenamiento y prueba
    # Mantenemos shuffle=False por buenas prácticas con datos de series de tiempo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, stratify=None)
    print(f"Datos de entrenamiento: {len(X_train)}, Datos de prueba: {len(X_test)}")

    # 4. Entrenar el modelo
    # Ya no es necesario class_weight='balanced' porque el dataset está balanceado
    print("\n⏳ Entrenando el modelo de FILTRO...")
    model = RandomForestClassifier(
        n_estimators=150, # Usamos un poco más de árboles
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("✅ Modelo de filtro entrenado.")

    # 5. Evaluar el modelo
    print("\n--- 📊 Reporte de Evaluación del Modelo de Filtro ---")
    predictions = model.predict(X_test)

    print("Matriz de Confusión:")
    #       Predijo Perdedora   Predijo Ganadora
    # Real P [[TN               FP              ]]
    # Real G [[FN               TP              ]]
    # TN: True Negative, FP: False Positive, FN: False Negative, TP: True Positive
    print(confusion_matrix(y_test, predictions))

    print("\nReporte de Clasificación:")
    # '0' es Perdedora (Loser), '1' es Ganadora (Winner)
    print(classification_report(y_test, predictions, target_names=['Loser (0)', 'Winner (1)']))

    # 6. Guardar el modelo de filtro final
    model_filename = 'trading_filter_model.joblib'
    joblib.dump(model, model_filename)
    print(f"\n✅ Modelo de FILTRO guardado como '{model_filename}'")