import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score, classification_report
from sqlalchemy import text
from db import conectar_db
from fastapi import HTTPException
import joblib
import pickle
import os

def obtener_datos_para_entrenamiento():
    conn = conectar_db()
    cursor = conn.cursor(dictionary=True)
    query = """
    SELECT 
        c.id AS cliente_id,
        c.edad,
        COUNT(DISTINCT p.id) AS cantidad_total_pedidos,
        MAX(p.fecha) AS fecha_ultima_compra,
        DATEDIFF(CURDATE(), MAX(p.fecha)) AS dias_desde_ultima_compra,
        SUM(dp.cantidad * dp.precio_unitario - dp.descuento) AS total_gastado,
        AVG((dp.cantidad * dp.precio_unitario - dp.descuento)) AS promedio_por_item
    FROM clientes c
    JOIN pedidos p ON c.id = p.cliente_id
    JOIN detalle_pedido dp ON p.id = dp.pedido_id
    GROUP BY c.id, c.edad
    """
    cursor.execute(query)
    datos = cursor.fetchall()
    cursor.close()
    conn.close()
    return datos

def _get_info_por_rango(date_from: str = None, date_to: str = None) -> tuple[str, tuple]:
    filtro = []
    params = []

    if date_from:
        filtro.append("p.fecha >= %s")
        params.append(date_from)
    if date_to:
        filtro.append("p.fecha <= %s")
        params.append(date_to)

    where = ""
    if filtro:
        where = "WHERE " + " AND ".join(filtro)

    sql = f"""
    SELECT 
        c.id AS cliente_id,
        c.edad,
        COUNT(DISTINCT p.id) AS cantidad_total_pedidos,
        DATEDIFF(CURDATE(), MAX(p.fecha)) AS dias_desde_ultima_compra,
        SUM(dp.cantidad * dp.precio_unitario - dp.descuento) AS total_gastado
    FROM clientes c
    JOIN pedidos p ON c.id = p.cliente_id
    JOIN detalle_pedido dp ON p.id = dp.pedido_id
    {where}
    GROUP BY c.id, c.edad
    """
    return sql, tuple(params)

def entrenar_modelo_regresion_lineal_old():
    datos = obtener_datos_para_entrenamiento()
    df = pd.DataFrame(datos)

    # Features
    X = df[['edad', 'cantidad_total_pedidos', 'dias_desde_ultima_compra', 'total_gastado']]
    
    # Target: valor promedio de un ítem comprado, como aproximación
    y = df['promedio_por_item']

    modelo = LinearRegression()
    modelo.fit(X, y)

    joblib.dump(modelo, 'modelos/modelo_lineal.pkl')
    print("Modelo entrenado y guardado en modelos/modelo_lineal.pkl")

def entrenar_modelo_regresion_lineal(date_from=None, date_to=None):
    engine = conectar_db()
    sql, params = _get_info_por_rango(date_from, date_to)
    df = pd.read_sql(sql, con=engine, params=params)

    df["volvera_comprar"] = df["dias_desde_ultima_compra"].apply(lambda x: 1 if x < 60 else 0)

    X = df[['edad', 'cantidad_total_pedidos', 'dias_desde_ultima_compra', 'total_gastado']]
    y = df['volvera_comprar']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    print(f"R2: {r2_score(y_test, y_pred):.3f}") # puntuación de regresión
    print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.3f}") # raíz del error cuadrático medio

    os.makedirs("modelos", exist_ok=True)
    with open("modelos/modelo_lineal.pkl", "wb") as f:
        pickle.dump(modelo, f)

    return {"mensaje": "Modelo lineal entrenado y guardado correctamente."}

def entrenar_modelo_regresion_logistica(date_from=None, date_to=None):
    engine = conectar_db()
    sql, params = _get_info_por_rango(date_from, date_to)
    df = pd.read_sql(sql, con=engine, params=params)

    # Se simula una columna binaria (por ejemplo: compró hace menos de 60 días o no)
    df["volvera_comprar"] = df["dias_desde_ultima_compra"].apply(lambda x: 1 if x < 60 else 0)

    X = df[['edad', 'cantidad_total_pedidos', 'dias_desde_ultima_compra', 'total_gastado']]
    y = df['volvera_comprar']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs("modelos", exist_ok=True)
    with open("modelos/modelo_logistico.pkl", "wb") as f:
        pickle.dump(modelo, f)

    return {"mensaje": "Modelo de clasificación entrenado y guardado correctamente."}

def entrenar_modelo_arbol_decision(date_from=None, date_to=None):
    engine = conectar_db()
    sql, params = _get_info_por_rango(date_from, date_to)
    df = pd.read_sql(sql, con=engine, params=params)

    df["volvera_comprar"] = df["dias_desde_ultima_compra"].apply(lambda x: 1 if x < 60 else 0)

    X = df[['edad', 'cantidad_total_pedidos', 'dias_desde_ultima_compra', 'total_gastado']]
    y = df['volvera_comprar']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)

    os.makedirs("modelos", exist_ok=True)
    with open("modelos/modelo_arbol.pkl", "wb") as f:
        pickle.dump(modelo, f)

    return {"mensaje": "Modelo de árbol entrenado y guardado correctamente."}

def entrenar_modelo_bosque_aleatorio(date_from=None, date_to=None):
    engine = conectar_db()
    sql, params = _get_info_por_rango(date_from, date_to)
    df = pd.read_sql(sql, con=engine, params=params)

    df["volvera_comprar"] = df["dias_desde_ultima_compra"].apply(lambda x: 1 if x < 60 else 0)

    X = df[['edad', 'cantidad_total_pedidos', 'dias_desde_ultima_compra', 'total_gastado']]
    y = df['volvera_comprar']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    os.makedirs("modelos", exist_ok=True)
    with open("modelos/modelo_bosque.pkl", "wb") as f:
        pickle.dump(modelo, f)

    return {"mensaje": "Modelo de bosque entrenado y guardado correctamente."}

def obtener_clientes_para_proyeccion():
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
    return df.to_dict(orient="records")

def predecir_con_modelo_lineal(edad, cantidad_total_pedidos, dias_desde_ultima_compra, total_gastado):
    modelo = joblib.load('modelos/modelo_lineal.pkl')

    features = [[edad, cantidad_total_pedidos, dias_desde_ultima_compra, total_gastado]]
    prediccion = modelo.predict(features)

    return {"valor_estimado": float(prediccion[0])}

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
