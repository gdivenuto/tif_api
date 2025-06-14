import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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

def forecast_demanda_mensual(
    n_periodos: int = 12,
    date_from: str | None = None,
    date_to:   str | None = None
) -> dict:
    """
    Genera un forecast de demanda mensual usando RandomForestRegressor.

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

    # Construir filtros de rango de fechas
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
    df['mes'] = pd.to_datetime(df['mes'])
    df.sort_values('mes', inplace=True)

    # Crear features temporales
    df['year'] = df['mes'].dt.year
    df['month_num'] = df['mes'].dt.month

    X = df[['year', 'month_num']]
    y = df['demanda']

    # Entrenar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Índices futuros
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

    # Predecir demanda futura
    y_future = model.predict(Xf)

    # Formatear resultados
    historico = [
        {"mes": dt.strftime('%Y-%m'), "demanda": float(d)}
        for dt, d in zip(df['mes'], df['demanda'])
    ]
    forecast = [
        {"mes": dt.strftime('%Y-%m'), "demanda": float(d)}
        for dt, d in zip(future_idx, y_future)
    ]

    # Guardar modelo
    os.makedirs("modelos", exist_ok=True)
    with open("modelos/modelo_demanda.pkl", "wb") as f:
        pickle.dump(model, f)

    return {"historico": historico, "forecast": forecast}

def consumo_material_mensual() -> dict:
    """
    Retorna datos para un gráfico de líneas con el consumo mensual de cada materia prima.
    """
    engine = conectar_db()
    sql = text("""
        SELECT 
          DATE_FORMAT(c.fecha, '%Y-%m') AS mes,
          dc.materia_prima_id,
          SUM(dc.cantidad) AS total_unidades
        FROM compras c
        JOIN detalle_compra dc ON c.id = dc.compra_id
        GROUP BY mes, dc.materia_prima_id
        ORDER BY mes
    """)
    df = pd.read_sql(sql, con=engine)
    pivot = df.pivot(index="mes", columns="materia_prima_id", values="total_unidades").fillna(0)

    return {
        "labels": pivot.index.tolist(),
        "datasets": [
            {"label": f"MP {mp_id}", "data": pivot[mp_id].tolist()}
            for mp_id in pivot.columns
        ]
    }

def gasto_por_proveedor() -> dict:
    """
    Retorna datos para un gráfico de pastel con la proporción de gasto por proveedor.
    """
    engine = conectar_db()
    sql = text("""
        SELECT 
          c.proveedor_id,
          SUM(dc.cantidad * dc.precio_unitario) AS gasto
        FROM compras c
        JOIN detalle_compra dc ON c.id = dc.compra_id
        GROUP BY c.proveedor_id
    """)
    df = pd.read_sql(sql, con=engine)
    return {
        "labels": df["proveedor_id"].tolist(),
        "data":   df["gasto"].tolist()
    }

def top_materiales() -> dict:
    """
    Retorna datos para un gráfico de barras con los 10 materiales más comprados (unidades).
    """
    engine = conectar_db()
    sql = text("""
        SELECT 
          dc.materia_prima_id,
          SUM(dc.cantidad) AS total_unidades
        FROM detalle_compra dc
        GROUP BY dc.materia_prima_id
        ORDER BY total_unidades DESC
        LIMIT 10
    """)
    df = pd.read_sql(sql, con=engine)
    return {
        "labels": df["materia_prima_id"].tolist(),
        "data":   df["total_unidades"].tolist()
    }

def color_usage() -> dict:
    """
    Retorna datos para un gráfico de doughnut con el uso porcentual de cada color.
    """
    engine = conectar_db()
    sql = text("""
        SELECT 
          dc.color_id,
          COUNT(*) AS usos
        FROM detalle_compra dc
        GROUP BY dc.color_id
    """)
    df = pd.read_sql(sql, con=engine)
    total = df["usos"].sum()
    percent = (df["usos"] / total * 100).round(1)
    return {
        "labels": df["color_id"].tolist(),
        "data":   df["usos"].tolist(),
        "percent": percent.tolist()
    }

