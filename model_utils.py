import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from db import conectar_db
from fastapi import HTTPException
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

def entrenar_modelo_regresion_lineal():
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

def entrenar_modelo_regresion_logistica():
    conn = conectar_db()
    query = """
        SELECT 
            c.id AS cliente_id,
            c.edad,
            COUNT(DISTINCT p.id) AS cantidad_total_pedidos,
            DATEDIFF(CURDATE(), MAX(p.fecha)) AS dias_desde_ultima_compra,
            SUM(dp.cantidad * dp.precio_unitario - dp.descuento) AS total_gastado
        FROM clientes c
        JOIN pedidos p ON c.id = p.cliente_id
        JOIN detalle_pedido dp ON p.id = dp.pedido_id
        GROUP BY c.id, c.edad
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # Simulamos columna binaria (por ejemplo: compró hace menos de 60 días)
    df["volvera_comprar"] = df["dias_desde_ultima_compra"].apply(lambda x: 1 if x < 60 else 0)

    X = df[['edad', 'cantidad_total_pedidos', 'dias_desde_ultima_compra', 'total_gastado']]
    y = df['volvera_comprar']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Guardar modelo
    os.makedirs("modelos", exist_ok=True)
    with open("modelos/modelo_logistico.pkl", "wb") as f:
        pickle.dump(modelo, f)

    return {"mensaje": "Modelo de clasificación entrenado y guardado correctamente."}

def entrenar_modelo_arbol_decision():
    conn = conectar_db()
    query = """
        SELECT 
            c.id AS cliente_id,
            c.edad,
            COUNT(DISTINCT p.id) AS cantidad_total_pedidos,
            DATEDIFF(CURDATE(), MAX(p.fecha)) AS dias_desde_ultima_compra,
            SUM(dp.cantidad * dp.precio_unitario - dp.descuento) AS total_gastado
        FROM clientes c
        JOIN pedidos p ON c.id = p.cliente_id
        JOIN detalle_pedido dp ON p.id = dp.pedido_id
        GROUP BY c.id, c.edad
    """
    df = pd.read_sql(query, conn)
    conn.close()

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

def entrenar_modelo_bosque_aleatorio():
    conn = conectar_db()
    query = """
        SELECT 
            c.id AS cliente_id,
            c.edad,
            COUNT(DISTINCT p.id) AS cantidad_total_pedidos,
            DATEDIFF(CURDATE(), MAX(p.fecha)) AS dias_desde_ultima_compra,
            SUM(dp.cantidad * dp.precio_unitario - dp.descuento) AS total_gastado
        FROM clientes c
        JOIN pedidos p ON c.id = p.cliente_id
        JOIN detalle_pedido dp ON p.id = dp.pedido_id
        GROUP BY c.id, c.edad
    """
    df = pd.read_sql(query, conn)
    conn.close()

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

def predecir_con_modelo_lineal(edad, cantidad_total_pedidos, dias_desde_ultima_compra, total_gastado):
    modelo = joblib.load('modelos/modelo_lineal.pkl')

    features = [[edad, cantidad_total_pedidos, dias_desde_ultima_compra, total_gastado]]
    prediccion = modelo.predict(features)

    return {"valor_estimado": float(prediccion[0])}

def predecir_por_cliente_id(cliente_id: int):
    conn = conectar_db()
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT 
        c.edad,
        COUNT(DISTINCT p.id) AS cantidad_total_pedidos,
        DATEDIFF(CURDATE(), MAX(p.fecha)) AS dias_desde_ultima_compra,
        SUM(dp.cantidad * dp.precio_unitario - dp.descuento) AS total_gastado
    FROM clientes c
    JOIN pedidos p ON c.id = p.cliente_id
    JOIN detalle_pedido dp ON p.id = dp.pedido_id
    WHERE c.id = %s
    GROUP BY c.id, c.edad
    """

    cursor.execute(query, (cliente_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Cliente no encontrado o sin datos suficientes")

    resultado = predecir_con_modelo_lineal(
        int(row['edad']),
        int(row['cantidad_total_pedidos']),
        int(row['dias_desde_ultima_compra']),
        float(row['total_gastado'])
    )

    return {
        "cliente_id": cliente_id,
        "resultado": round(resultado, 2)
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

def obtener_clientes_para_proyeccion():
    conn = conectar_db()
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
    df = pd.read_sql(query, conn)
    conn.close()
    return df.to_dict(orient="records")
