import pandas as pd
import numpy as np
import os
import pickle

from typing import Optional
from fastapi import Query
from sqlalchemy import text
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  (
    r2_score, # Coeficiente de determinación, función de puntuación de regresión R^2
    mean_squared_error, # Error cuadrático medio
    root_mean_squared_error, # Raíz del error cuadrático medio
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from db import conectar_db

def entrenar_modelo_demanda_mensual_mp(
    n_periodos: int = Query(12, description="Número de meses a predecir"),
    date_from: Optional[str] = None,
    date_to:   Optional[str] = None
) -> dict:
    """
    Genera un forecast de demanda mensual optimizado usando RandomForestRegressor
    con características de series temporales (lags y estacionalidad cíclica).

    Args:
        n_periodos (int): Número de meses futuros a predecir.
        date_from (str, opcional): Fecha mínima 'YYYY-MM-DD' para filtrar el histórico.
        date_to   (str, opcional): Fecha máxima 'YYYY-MM-DD' para filtrar el histórico.

    Returns:
        dict: {
            "historico": List[{"mes": "YYYY-MM", "demanda": float}],
            "forecast":  List[{"mes": "YYYY-MM", "demanda": float}],
            "r2": float,
            "mse": float,
            "rmse": float
        }
    """
    engine = conectar_db()

    # Filtros opcionales
    filtros: list[str] = []
    params: dict[str, str] = {}
    if date_from:
        filtros.append("c.fecha >= :date_from")
        params["date_from"] = date_from
    if date_to:
        filtros.append("c.fecha <= :date_to")
        params["date_to"] = date_to
    where_clause = f"WHERE {' AND '.join(filtros)}" if filtros else ""

    # Consulta de demanda mensual
    sql = text(f"""
        SELECT
            DATE_FORMAT(c.fecha, '%Y-%m-01') AS mes,
            SUM(dc.cantidad) AS demanda
        FROM compras c
        JOIN detalle_compra dc ON c.id = dc.compra_id
        {where_clause}
        GROUP BY mes
        ORDER BY mes
    """
    )

    df = pd.read_sql(sql, con=engine, params=params)
    if df.empty or len(df) < n_periodos:
        return {"historico": [], "forecast": [], "r2": None, "rmse": None}

    df['mes'] = pd.to_datetime(df['mes'])
    df.sort_values('mes', inplace=True)

    # Se crean características de series temporales
    df['year'] = df['mes'].dt.year
    df['month'] = df['mes'].dt.month
    # Estacionalidad cíclica
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    # Lags
    df['lag_1'] = df['demanda'].shift(1)
    df['lag_12'] = df['demanda'].shift(12)

    df = df.dropna().reset_index(drop=True)

    if len(df) <= n_periodos:
        return {
            "mensaje": "No hay datos suficientes tras procesar las características de series temporales.",
            "historico": [],
            "forecast": [],
            "r2": None,
            "rmse": None
        }

    # Se separa en histórico y en test temporal
    train = df.iloc[:-n_periodos]
    test  = df.iloc[-n_periodos:]

    features = ['year', 'month', 'sin_month', 'cos_month', 'lag_1', 'lag_12']
    X_train, y_train = train[features], train['demanda']
    X_test,  y_test  = test[features],  test['demanda']

    # Pipeline con escalado y modelo
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Métricas
    r2   = r2_score(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    # Forecast futuros
    last_month = df['mes'].max()
    future_idx = pd.date_range(
        start = last_month + pd.offsets.MonthBegin(),
        periods = n_periodos,
        freq='MS'
    )
    future = pd.DataFrame({
        'year': future_idx.year,
        'month': future_idx.month
    })
    future['sin_month'] = np.sin(2 * np.pi * future['month'] / 12)
    future['cos_month'] = np.cos(2 * np.pi * future['month'] / 12)
    # Para lag_1 y lag_12 se toman del test final
    future['lag_1'] = list(test['demanda'].values)  # lag1 primer valor
    future['lag_12'] = list(df['demanda'].shift(12).iloc[-n_periodos:].values)

    y_future = pipeline.predict(future[features])

    historico = [
        {"mes": dt.strftime('%Y-%m'), "demanda": float(d)}
        for dt, d in zip(train['mes'], train['demanda'])
    ] + [
        {"mes": dt.strftime('%Y-%m'), "demanda": float(d)}
        for dt, d in zip(test['mes'], test['demanda'])
    ]
    forecast = [
        {"mes": dt.strftime('%Y-%m'), "demanda": float(d)}
        for dt, d in zip(future_idx, y_future)
    ]

    # Se guarda el modelo
    os.makedirs('modelos', exist_ok=True)
    with open('modelos/modelo_demanda_mensual_mp.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    return {
        "historico": historico, 
        "forecast": forecast, 
        "r2": round(r2,3), 
        "mse": round(mse,3),
        "rmse": round(rmse,3)
    }
