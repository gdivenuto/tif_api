import math
import pickle
import os
import pandas as pd

from db import conectar_db

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
        ORDER BY materia_prima_nombre
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
