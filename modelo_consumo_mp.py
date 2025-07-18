import math
import pickle
import os
import pandas as pd

from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, # Coeficiente de determinación: función de puntuación de regresión R^2
    mean_squared_error, # Error cuadrático medio
    root_mean_squared_error, # Raíz del error cuadrático medio
    mean_absolute_error, # Error absoluto medio
    mean_absolute_percentage_error, # Error porcentual absoluto medio
    median_absolute_error, # Desviación absoluta mediana
)

from db import conectar_db

def entrenar_modelo_consumo_materia_prima(
    date_from: Optional[str] = None,
    date_to:   Optional[str] = None
) -> dict:
    """
    Entrena un RandomForestRegressor para estimar el consumo mensual
    de cada materia prima, usando datos históricos de compras.

    Args:
        date_from (str, optional): Fecha mínima 'YYYY-MM-DD' para filtrar histórico.
        date_to   (str, optional): Fecha máxima 'YYYY-MM-DD' para filtrar histórico.

    Returns:
        dict: {
            "mensaje": str,
            "r2": float,    # coeficiente de determinación
            "mse": float,   # error cuadrático medio
            "rmse": float   # raíz del error cuadrático medio
        }
        o un mensaje en caso de no haber datos.
    """
    engine = conectar_db()

    sql = """
        SELECT
          DATE_FORMAT(c.fecha, '%%Y-%%m-01') AS mes,
          dc.materia_prima_id,
          SUM(dc.cantidad) AS consumo
        FROM compras c
        JOIN detalle_compra dc ON c.id = dc.compra_id
        WHERE (%s IS NULL OR c.fecha >= %s)
          AND (%s IS NULL OR c.fecha <= %s)
        GROUP BY mes, dc.materia_prima_id
        ORDER BY mes
    """
    params = (date_from, date_from, date_to, date_to)
    df = pd.read_sql(sql, con=engine, params=params)

    if df.empty or len(df) < 2:
        return {"mensaje": "No hay datos suficientes para entrenar el modelo de consumo."}

    df["mes"]   = pd.to_datetime(df["mes"])
    df["year"]  = df["mes"].dt.year
    df["month"] = df["mes"].dt.month

    X = df[["year", "month", "materia_prima_id"]]
    y = df["consumo"]

    try:
        # Se define un pipeline
        numeric_features = ["year", "month"]
        cat_features     = ["materia_prima_id"]
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
        ])
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("rf", RandomForestRegressor(random_state=42))
        ])

        # Split y entrenamiento
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        # modelo.fit(X_train, y_train)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        #y_pred = modelo.predict(X_test)

        # Se obtienen métricas
        r2    = r2_score(y_test, y_pred)
        mse   = mean_squared_error(y_test, y_pred)
        rmse  = root_mean_squared_error(y_test, y_pred)
        mae   = mean_absolute_error(y_test, y_pred)
        mape  = mean_absolute_percentage_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)

        # Se guarda el modelo
        os.makedirs("modelos", exist_ok=True)
        with open("modelos/modelo_consumo_mp.pkl", "wb") as f:
            pickle.dump(pipeline, f) # se reemplazó modelo por el pipeline

        return {
            "mensaje": "El modelo de consumo de materias primas se ha entrenado y guardado correctamente.",
            "r2": round(r2, 3),
            "mse": round(mse, 3),
            "rmse": round(rmse, 3),
            "mae":   round(mae,   3),
            "mape":  round(mape,  3),
            "medae": round(medae, 3)
        }
    except Exception as e:
        return {
            "mensaje": "Error al entrenar o guardar el modelo de consumo.",
            "error": str(e)
        }

def predecir_consumo_materia_prima() -> dict:
    """
    Carga el modelo de consumo entrenado y predice la demanda del mes siguiente
    para cada materia prima, devolviendo también su nombre.
    """
    modelo_path = "modelos/modelo_consumo_mp.pkl"
    if not os.path.exists(modelo_path):
        return {"forecast": []}

    with open(modelo_path, "rb") as f:
        modelo = pickle.load(f)

    engine = conectar_db()

    # Se obtiene la última fecha de compra
    last_fecha = pd.read_sql("SELECT MAX(fecha) AS last FROM compras", con=engine)["last"][0]
    if not last_fecha:
        return {"forecast": []}

    last = pd.to_datetime(last_fecha)
    next_month = (last + pd.offsets.MonthBegin()).replace(day=1)
    mes_str = next_month.strftime("%Y-%m")

    # Se obtiene el id, codigo y nombre de cada materia prima comprada
    mp_df = pd.read_sql("""
        SELECT DISTINCT
          dc.materia_prima_id,
          mp.codigo AS materia_prima_codigo,
          mp.nombre AS materia_prima_nombre
        FROM detalle_compra dc
        JOIN materias_primas mp
          ON dc.materia_prima_id = mp.id
    """, con=engine)

    if mp_df.empty:
        return {"forecast": []}

    # Se preparan los features de predicción
    Xf = pd.DataFrame({
        "year": [next_month.year] * len(mp_df),
        "month": [next_month.month] * len(mp_df),
        "materia_prima_id": mp_df["materia_prima_id"]
    })

    # Se predice
    preds = modelo.predict(Xf)

    # Se construye la lista con la información a mostrar en el sistema
    forecast = []
    for (idx, row), p in zip(mp_df.iterrows(), preds):
        forecast.append({
            "mes": mes_str,
            "materia_prima_id": int(row["materia_prima_id"]),
            "materia_prima_codigo": row["materia_prima_codigo"],
            "materia_prima_nombre": row["materia_prima_nombre"],
            "consumo_estimado": math.ceil(p) #round(float(p), 2)
        })

    return {"forecast": forecast}
