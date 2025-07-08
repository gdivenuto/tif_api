import pickle
import os
import pandas as pd

from typing import Optional
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import text
from db import conectar_db

def forecast_demanda_mensual(
    n_periodos: int = 12,
    date_from: Optional[str] = None,
    date_to:   Optional[str] = None
) -> dict:
    """
    Genera un pronóstico de demanda mensual usando RandomForestRegressor.
    Si no hay datos históricos, devuelve historico=[] y forecast=[].

    Args:
        n_periodos (int): Número de meses futuros a predecir.
        date_from (str, opcional): Fecha mínima 'YYYY-MM-DD' para filtrar el histórico.
        date_to   (str, opcional): Fecha máxima 'YYYY-MM-DD' para filtrar el histórico.

    Returns:
        dict: {
            "historico": List[{"mes": "YYYY-MM", "demanda": float}],
            "forecast":   List[{"mes": "YYYY-MM", "demanda": float}]
        }
    """
    engine = conectar_db()

    filtros, params = [], {}
    if date_from:
        filtros.append("c.fecha >= :date_from")
        params["date_from"] = date_from
    if date_to:
        filtros.append("c.fecha <= :date_to")
        params["date_to"] = date_to
    where = f"WHERE {' AND '.join(filtros)}" if filtros else ""

    sql = text(f"""
        SELECT
            DATE_FORMAT(c.fecha, '%Y-%m-01') AS mes,
            SUM(dc.cantidad) AS demanda
        FROM compras c
        JOIN detalle_compra dc ON c.id = dc.compra_id
        {where}
        GROUP BY mes
        ORDER BY mes
    """)
    df = pd.read_sql(sql, con=engine, params=params)

    if df.empty:
        return {"historico": [], "forecast": []}

    df['mes'] = pd.to_datetime(df['mes'])
    df.sort_values('mes', inplace=True)

    df['year'] = df['mes'].dt.year
    df['month_num'] = df['mes'].dt.month

    X = df[['year', 'month_num']]
    y = df['demanda']

    # Se entrena el modelo, con 100 estimadores (árboles de decisión)
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X, y)

    last_month = df['mes'].max()
    future_idx = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(),
        periods=n_periodos,
        freq='MS'
    )
    Xf = pd.DataFrame({
        'year': future_idx.year,
        'month_num': future_idx.month
    })

    # Se predice la demanda futura
    y_future = modelo.predict(Xf)

    historico = [
        {
            "mes": dt.strftime('%Y-%m'), 
            "demanda": float(d)
        }
        for dt, d in zip(df['mes'], df['demanda'])
    ]
    forecast = [
        {
            "mes": dt.strftime('%Y-%m'), 
            "demanda": float(d)
        }
        for dt, d in zip(future_idx, y_future)
    ]

    # Se guarda el modelo
    os.makedirs("modelos", exist_ok=True)
    with open("modelos/modelo_demanda.pkl", "wb") as f:
        pickle.dump(modelo, f)

    return {"historico": historico, "forecast": forecast}
