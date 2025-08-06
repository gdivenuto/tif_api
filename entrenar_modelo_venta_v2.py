import pickle
import os
import pandas as pd
from typing import Optional

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, # Coeficiente de determinación, función de puntuación de regresión R^2
    mean_squared_error, # Error cuadrático medio
    root_mean_squared_error, # Raíz del error cuadrático medio
    mean_absolute_error, # Error absoluto medio
    mean_absolute_percentage_error, # Error porcentual absoluto medio
    median_absolute_error, # Desviación absoluta mediana
)

from db import conectar_db

def _get_info_ventas(date_from: Optional[str], date_to: Optional[str]) -> tuple[str, tuple]:
    filtro = []
    params = []

    if date_from:
        filtro.append("p.fecha >= %s")
        params.append(date_from)
    if date_to:
        filtro.append("p.fecha <= %s")
        params.append(date_to)
    
    where = f"WHERE {' AND '.join(filtro)}" if filtro else ""

    sql = f"""
    SELECT 
        c.id AS cliente_id,
        COUNT(DISTINCT p.id) AS cantidad_total_pedidos,
        DATEDIFF(CURDATE(), MAX(p.fecha)) AS dias_desde_ultima_compra,
        SUM(dp.cantidad * dp.precio_unitario - dp.descuento) AS total_gastado
    FROM clientes c
    JOIN pedidos p ON c.id = p.cliente_id
    JOIN detalle_pedido dp ON p.id = dp.pedido_id
    {where}
    GROUP BY c.id , c.edad
    """
    return sql, tuple(params)

def entrenar_modelo_venta_regresion_lineal(
    date_from: Optional[str] = None,
    date_to:   Optional[str] = None
) -> dict:
    """
    Entrena un LinearRegression para estimar el total gastado de cada cliente (variable continua) 
    A partir de:
      - cantidad_total_pedidos
      - dias_desde_ultima_compra

    Args:
        date_from (str, optional): fecha mínima 'YYYY-MM-DD' para filtrar histórico.
        date_to   (str, optional): fecha máxima 'YYYY-MM-DD' para filtrar histórico.

    Returns:
        dict: {
          "mensaje": str,
          "r2": float,    # Coeficiente de determinación
          "mse": float,   # Error Cuadrático Medio
          "rmse": float,  # Raíz del Error Cuadrático Medio
          "mae": float,   # Error Medio Absoluto
          "mape": float,  # Error Porcentual Absoluto Medio
          "medae": float, # Mediana del Error Absoluto
        }
        o un mensaje en caso de no haber datos.
    """
    engine = conectar_db()
    sql, params = _get_info_ventas(date_from, date_to)
    df = pd.read_sql(sql, con=engine, params=params)

    # Se verifica que haya datos
    if df.empty:
        return {"mensaje": "No hay datos suficientes para entrenar el modelo lineal de ventas."}

    # Features
    X = df[['cantidad_total_pedidos', 'dias_desde_ultima_compra']]
    # Target
    y = df['total_gastado']

    # Se verifica el tamaño mínimo para la división de datos de entrenamiento y de prueba
    if len(df) < 2:
        return {"mensaje": "Se requieren al menos 2 registros para entrenar y validar el modelo."}

    try:
        # Se dividen
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Se entrena
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

        # Se predice
        y_pred = modelo.predict(X_test)

        # Se obtienen métricas
        r2    = r2_score(y_test, y_pred)
        mse   = mean_squared_error(y_test, y_pred)
        rmse  = root_mean_squared_error(y_test, y_pred) # math.sqrt(mse)
        mae   = mean_absolute_error(y_test, y_pred)
        mape  = mean_absolute_percentage_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)

        # Se guarda el modelo
        os.makedirs("modelos", exist_ok=True)
        with open("modelos/modelo_lineal_ventas_v2.pkl", "wb") as f:
            pickle.dump(modelo, f)
        
        # Se devuelve la confirmación y se incluyen métricas
        return {
            "mensaje": "Modelo lineal de ventas entrenado y guardado correctamente.",
            "r2": round(r2, 3),
            "mse": round(mse, 3),
            "rmse": round(rmse, 3),
            "mae":   round(mae,   3),
            "mape":  round(mape,  3),
            "medae": round(medae, 3)
        }

    except Exception as e:
        return {
            "mensaje": "Error al entrenar o guardar el modelo de ventas.",
            "error": str(e)
        }
