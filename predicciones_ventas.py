import joblib
import pickle
import os
import pandas as pd

from sqlalchemy import text
from fastapi import HTTPException

from db import conectar_db

# Retorna los Clientes que compraron, con su monto total gastado
def obtener_clientes_que_compraron():
    engine = conectar_db()
    query = """
        SELECT 
            c.id AS id,
            CONCAT(c.nombre, ' ', c.apellido) AS nombre,
            c.edad,
            COUNT(DISTINCT p.id) AS pedidos,
            DATEDIFF(CURDATE(), MAX(p.fecha)) AS dias,
            SUM(dp.cantidad * dp.precio_unitario - dp.descuento) AS gastado
        FROM clientes c
        JOIN pedidos p ON c.id = p.cliente_id
        JOIN detalle_pedido dp ON p.id = dp.pedido_id
        GROUP BY c.id, c.nombre, c.apellido, c.edad
    """
    df = pd.read_sql(query, con=engine)

    if df.empty:
        return []
    
    return df.to_dict(orient="records")

# Se predice utilizando el modelo respectivo a cada modelo algoritmo
def predecir_con_modelo_lineal(edad, cantidad_total_pedidos, dias_desde_ultima_compra, total_gastado):
    modelo = joblib.load('modelos/modelo_lineal.pkl')

    features = [[edad, cantidad_total_pedidos, dias_desde_ultima_compra, total_gastado]]
    prediccion = modelo.predict(features)

    return {"valor_estimado": float(prediccion[0])}

def predecir_con_modelo_logistico(edad, cantidad_total_pedidos, dias_desde_ultima_compra, total_gastado):
    modelo_path = "modelos/modelo_logistico.pkl"
    if not os.path.exists(modelo_path):
        raise FileNotFoundError("El modelo de clasificación no está entrenado.")

    with open(modelo_path, "rb") as f:
        modelo = pickle.load(f)

    X = [[edad, cantidad_total_pedidos, dias_desde_ultima_compra, total_gastado]]
    pred = modelo.predict(X)[0]
    prob = modelo.predict_proba(X)[0][1]  # Probabilidad de que sea clase 1

    return {
        "volvera_comprar": bool(pred),
        "probabilidad": round(prob, 2)
    }

def predecir_con_modelo_arbol(edad, cantidad_total_pedidos, dias_desde_ultima_compra, total_gastado):
    modelo_path = "modelos/modelo_arbol.pkl"
    if not os.path.exists(modelo_path):
        raise FileNotFoundError("El modelo de árbol no está entrenado.")
    with open(modelo_path, "rb") as f:
        modelo = pickle.load(f)

    X = [[edad, cantidad_total_pedidos, dias_desde_ultima_compra, total_gastado]]
    pred = modelo.predict(X)[0]
    prob = modelo.predict_proba(X)[0][1] if hasattr(modelo, 'predict_proba') else None

    return {
        "volvera_comprar": bool(pred),
        "probabilidad": round(prob, 2) if prob is not None else "no disponible"
    }

def predecir_con_modelo_bosque(edad, cantidad_total_pedidos, dias_desde_ultima_compra, total_gastado):
    modelo_path = "modelos/modelo_bosque.pkl"
    if not os.path.exists(modelo_path):
        raise FileNotFoundError("El modelo de bosque no está entrenado.")
    with open(modelo_path, "rb") as f:
        modelo = pickle.load(f)

    X = [[edad, cantidad_total_pedidos, dias_desde_ultima_compra, total_gastado]]
    pred = modelo.predict(X)[0]
    prob = modelo.predict_proba(X)[0][1]

    return {
        "volvera_comprar": bool(pred),
        "probabilidad": round(prob, 2)
    }

def predecir_por_cliente_id(cliente_id: int) -> dict:
    """
    Calcula la predicción de compra futura para un cliente dado por su Id

    Args:
        cliente_id (int): identificador único del cliente en la tabla `clientes`.

    Returns:
        dict: {
            "cliente_id": int,
            "valor_estimado": float  # monto proyectado de su próxima compra
        }

    Raises:
        HTTPException: si no se encuentra el cliente o no hay datos suficientes.
    """
    engine = conectar_db()
    
    query = text("""
        SELECT 
            c.edad AS edad,
            COUNT(DISTINCT p.id) AS cantidad_total_pedidos,
            DATEDIFF(CURDATE(), MAX(p.fecha)) AS dias_desde_ultima_compra,
            SUM(dp.cantidad * dp.precio_unitario - dp.descuento) AS total_gastado
        FROM clientes c
        JOIN pedidos p ON c.id = p.cliente_id
        JOIN detalle_pedido dp ON p.id = dp.pedido_id
        WHERE c.id = :cliente_id
        GROUP BY c.id, c.edad
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"cliente_id": cliente_id})
        row = result.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Cliente no encontrado o sin datos suficientes")

    # row es un RowMapping, se accede por nombre de columna
    edad          = int(row["edad"])
    pedidos       = int(row["cantidad_total_pedidos"])
    dias          = int(row["dias_desde_ultima_compra"])
    total_gastado = float(row["total_gastado"])

    # Se realiza la predicción lineal
    valor_estimado = predecir_con_modelo_lineal(edad, pedidos, dias, total_gastado)

    return {
        "cliente_id": cliente_id,
        "valor_estimado": round(valor_estimado, 2)
    }
