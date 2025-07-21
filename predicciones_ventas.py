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
            COUNT(DISTINCT p.id) AS cantidad_total_pedidos,
            DATEDIFF(CURDATE(), MAX(p.fecha)) AS dias_desde_ultima_compra,
            SUM(dp.cantidad * dp.precio_unitario - dp.descuento) AS total_gastado
        FROM clientes c
        JOIN pedidos p ON c.id = p.cliente_id
        JOIN detalle_pedido dp ON p.id = dp.pedido_id
        GROUP BY c.id, c.nombre, c.apellido, c.edad
    """
    df = pd.read_sql(query, con=engine)

    if df.empty:
        return []
    
    return df.to_dict(orient="records")

# Predice la venta futura de un cliente determinado por su ID
def predecir_ventas_futuras_por_cliente(cliente_id: int) -> dict:
    """
    Carga el modelo entrenado y predice cuánto gastará el cliente en base
    a sus datos agregados de historial.

    Args:
        cliente_id (int): id del cliente a predecir.

    Returns:
        dict: {
          "cliente_id": int,
          "estimado": float
        }
        o bien {"error": "..."} si falta modelo o datos.
    """
    modelo_path = "modelos/modelo_lineal_ventas.pkl"
    if not os.path.exists(modelo_path):
        return {"error": "Modelo de ventas no entrenado aún."}

    # Se carga el modelo
    with open(modelo_path, "rb") as f:
        modelo = pickle.load(f)

    # Obtener features del cliente
    engine = conectar_db()
    sql = """
    SELECT
        c.edad,
        COUNT(DISTINCT p.id) AS cantidad_total_pedidos,
        DATEDIFF(CURDATE(), MAX(p.fecha)) AS dias_desde_ultima_compra
    FROM clientes c
    JOIN pedidos p ON c.id = p.cliente_id
    WHERE c.id = %s
    GROUP BY c.id, c.edad
    """
    df = pd.read_sql(sql, con=engine, params=(cliente_id,))

    if df.empty:
        return {"error": "Cliente no encontrado o sin historial de ventas."}

    # Preparar array con nombres de columnas
    cols = ['edad', 'cantidad_total_pedidos', 'dias_desde_ultima_compra']
    X_new = df[cols]

    # Se predice
    pred = modelo.predict(X_new)[0]
    return {
        "cliente_id": cliente_id,
        "estimado": round(float(pred), 2)
    }

# Predice la venta futura para todos los clientes
def predecir_ventas_futuras() -> dict:
    """
    Carga el modelo lineal entrenado y predice el gasto futuro
    para todos los clientes de la base.

    Returns:
        dict: {
            "predicciones": [
                {"cliente_id": int, "estimado": float},
                …
            ]
        }
    """
    modelo_path = "modelos/modelo_lineal_ventas.pkl"
    if not os.path.exists(modelo_path):
        return {"predicciones": []}

    # Se carga el modelo
    with open(modelo_path, "rb") as f:
        modelo = pickle.load(f)

    engine = conectar_db()
    sql = """
    SELECT 
        c.id AS cliente_id,
        CONCAT(c.nombre, ' ', c.apellido) AS nombre,
        c.edad AS edad,
        COUNT(DISTINCT p.id) AS cantidad_total_pedidos,
        DATEDIFF(CURDATE(), MAX(p.fecha)) AS dias_desde_ultima_compra,
        SUM(dp.cantidad * dp.precio_unitario - dp.descuento) AS total_gastado
    FROM clientes c
    JOIN pedidos p ON c.id = p.cliente_id
    JOIN detalle_pedido dp ON p.id = dp.pedido_id
    GROUP BY c.id, c.nombre, c.apellido, c.edad
    ORDER BY nombre
    """
    df = pd.read_sql(text(sql), con=engine)

    if df.empty:
        return {"predicciones": []}

    # Se preparan las columnas tal como se entrenó el modelo
    cols = ['edad', 'cantidad_total_pedidos', 'dias_desde_ultima_compra']
    X = df[cols]

    # Se predice
    preds = modelo.predict(X)

    # Se arma la lista de resultados
    lista = []
    for cliente_id, nombre, pred in zip(df['cliente_id'], df['nombre'], preds):
        lista.append({
            "cliente_id": int(cliente_id),
            "nombre": nombre,
            "estimado": round(float(pred), 2)
        })

    return {"predicciones": lista}

# Se predice utilizando el modelo respectivo
# ------------------------------------------------------------------
def predecir_ventas_con_modelo_lineal(edad, cantidad_total_pedidos, dias_desde_ultima_compra, total_gastado):
    modelo = joblib.load('modelos/modelo_lineal_ventas.pkl')

    features = [[edad, cantidad_total_pedidos, dias_desde_ultima_compra, total_gastado]]
    prediccion = modelo.predict(features)

    return {"valor_estimado": float(prediccion[0])}

def predecir_ventas_con_modelo_logistico(edad, cantidad_total_pedidos, dias_desde_ultima_compra, total_gastado):
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

def predecir_ventas_con_modelo_arbol(edad, cantidad_total_pedidos, dias_desde_ultima_compra, total_gastado):
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

def predecir_ventas_con_modelo_bosque(edad, cantidad_total_pedidos, dias_desde_ultima_compra, total_gastado):
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
    valor_estimado = predecir_ventas_con_modelo_lineal(edad, pedidos, dias, total_gastado)

    return {
        "cliente_id": cliente_id,
        "valor_estimado": round(valor_estimado, 2)
    }
