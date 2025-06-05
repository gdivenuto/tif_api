import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from db import conectar_db
from fastapi import HTTPException

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

def entrenar_y_guardar_modelo():
    datos = obtener_datos_para_entrenamiento()
    df = pd.DataFrame(datos)

    # Features
    X = df[['edad', 'cantidad_total_pedidos', 'dias_desde_ultima_compra', 'total_gastado']]
    
    # Target: valor promedio de un ítem comprado, como aproximación
    y = df['promedio_por_item']

    modelo = LinearRegression()
    modelo.fit(X, y)

    joblib.dump(modelo, 'modelos/modelo.pkl')
    print("Modelo entrenado y guardado en modelos/modelo.pkl")

def predecir_con_modelo(edad, cantidad_total_pedidos, dias_desde_ultima_compra, total_gastado):
    modelo = joblib.load('modelos/modelo.pkl')

    features = [[edad, cantidad_total_pedidos, dias_desde_ultima_compra, total_gastado]]
    prediccion = modelo.predict(features)

    return float(prediccion[0])

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

    resultado = predecir_con_modelo(
        int(row['edad']),
        int(row['cantidad_total_pedidos']),
        int(row['dias_desde_ultima_compra']),
        float(row['total_gastado'])
    )

    return {
        "cliente_id": cliente_id,
        "resultado": round(resultado, 2)
    }

# Solo para pruebas, para ver la estructura de las tablas
def debug_estructura_detalle_pedido():
    conn = conectar_db()
    cursor = conn.cursor()
    cursor.execute("SHOW CREATE TABLE detalle_pedido")
    resultado = cursor.fetchone()
    print(resultado)
    cursor.close()
    conn.close()