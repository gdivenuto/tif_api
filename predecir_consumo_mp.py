import math
import pickle
import os
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error
)
from db import conectar_db

def predecir_consumo_materia_prima() -> dict:
    """
    Carga el modelo de consumo entrenado y:
      1) Predice la demanda del mes siguiente para cada materia prima.
      2) Vuelve a predecir sobre el histórico completo para obtener métricas.

    Returns:
        dict: {
            "forecast": [ {mes, materia_prima_id, materia_prima_codigo,
                           materia_prima_nombre, consumo_estimado}, … ],
            "metrics": {
               "r2":   float,
               "mse":  float,
               "rmse": float,
               "mae":  float,
               "mape": float,
               "medae":float
            }
        }
    """
    modelo_path = "modelos/modelo_consumo_mp.pkl"
    if not os.path.exists(modelo_path):
        return {"forecast": [], "metrics": {}}

    # Cargo modelo
    with open(modelo_path, "rb") as f:
        modelo = pickle.load(f)

    engine = conectar_db()

    # ------ 1) Forecast próximo mes ------
    last_fecha = pd.read_sql("SELECT MAX(fecha) AS last FROM compras", con=engine)["last"][0]
    if not last_fecha:
        return {"forecast": [], "metrics": {}}

    last = pd.to_datetime(last_fecha)
    next_month = (last + pd.offsets.MonthBegin()).replace(day=1)
    mes_str = next_month.strftime("%Y-%m")

    mp_df = pd.read_sql("""
        SELECT DISTINCT
          dc.materia_prima_id,
          mp.codigo   AS materia_prima_codigo,
          mp.nombre   AS materia_prima_nombre
        FROM detalle_compra dc
        JOIN materias_primas mp ON dc.materia_prima_id = mp.id
        ORDER BY mp.nombre
    """, con=engine)

    # Si no hay materias primas
    if mp_df.empty:
        return {"forecast": [], "metrics": {}}

    Xf = pd.DataFrame({
        "year":             [next_month.year]  * len(mp_df),
        "month":            [next_month.month] * len(mp_df),
        "materia_prima_id": mp_df["materia_prima_id"].tolist()
    })
    preds = modelo.predict(Xf)

    forecast = []
    for (_, row), p in zip(mp_df.iterrows(), preds):
        forecast.append({
            "mes":                  mes_str,
            "materia_prima_id":     int(row["materia_prima_id"]),
            "materia_prima_codigo": row["materia_prima_codigo"],
            "materia_prima_nombre": row["materia_prima_nombre"],
            "consumo_estimado":     math.ceil(p)
        })

    # ------ 2) Métricas sobre histórico ------
    hist = pd.read_sql("""
        SELECT
          DATE_FORMAT(c.fecha, '%%Y-%%m-01') AS mes,
          dc.materia_prima_id,
          SUM(dc.cantidad) AS consumo
        FROM compras c
        JOIN detalle_compra dc ON c.id = dc.compra_id
        GROUP BY mes, dc.materia_prima_id
        ORDER BY mes
    """, con=engine)

    # parsear y preparar X/y
    hist["mes"] = pd.to_datetime(hist["mes"])
    Xh = pd.DataFrame({
        "year":            hist["mes"].dt.year,
        "month":           hist["mes"].dt.month,
        "materia_prima_id":hist["materia_prima_id"]
    })
    yh = hist["consumo"].values
    ph = modelo.predict(Xh)

    # calcular métricas
    r2   = r2_score(yh, ph)
    mse  = mean_squared_error(yh, ph)
    rmse = math.sqrt(mse)
    mae  = mean_absolute_error(yh, ph)
    mape = mean_absolute_percentage_error(yh, ph)
    medae= median_absolute_error(yh, ph)

    metrics = {
        "r2":    round(r2,   3),
        "mse":   round(mse,  3),
        "rmse":  round(rmse, 3),
        "mae":   round(mae,   3),
        "mape":  round(mape,  3),
        "medae": round(medae, 3)
    }

    return {"forecast": forecast, "metrics": metrics}
