import joblib
import pickle
import os
import pandas as pd
from typing import Optional

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    root_mean_squared_error, # Pérdida de regresión del error cuadrático medio
    mean_squared_error, # Raíz del error cuadrático medio
    r2_score, # Coeficiente de determinación: función de puntuación de regresión R^2
    classification_report, # reporte con las principales métricas de clasificación
    accuracy_score, # proporción de aciertos global
    precision_score, # de los positivos predichos, cuántos son correctos
    recall_score, # sensibilidad de los positivos reales, cuántos se aciertan
    f1_score, # media armónica de precision y recall
)
from sqlalchemy import text
from fastapi import HTTPException

from db import conectar_db

def entrenar_modelo_consumo_materia_prima(
    date_from: Optional[str] = None,
    date_to:   Optional[str] = None
) -> dict:
    """
    Entrena un RandomForestRegressor para estimar el consumo mensual
    de cada materia prima, usando datos históricos de compras.

    Args:
        date_from (str, optional): Fecha mínima 'YYYY-MM-DD' para filtrar histórico.
        date_to   (str, optional): Fecha máxima 'YYYY-MM-DD' para filtrar histórico.

    Returns:
        dict: {
            "mensaje": str,
            "r2": float,    # coeficiente de determinación
            "rmse": float   # raíz del error cuadrático medio
        }
        o un mensaje en caso de no haber datos.
    """
    engine = conectar_db()

    sql = """
        SELECT
          DATE_FORMAT(c.fecha, '%Y-%m-01') AS mes,
          dc.materia_prima_id,
          SUM(dc.cantidad) AS consumo
        FROM compras c
        JOIN detalle_compra dc ON c.id = dc.compra_id
        WHERE (%s IS NULL OR c.fecha >= %s)
          AND (%s IS NULL OR c.fecha <= %s)
        GROUP BY mes, dc.materia_prima_id
        ORDER BY mes
    """
    params = (date_from, date_from, date_to, date_to)
    df = pd.read_sql(sql, con=engine, params=params)

    if df.empty:
        return {"mensaje": "No hay datos suficientes para entrenar el modelo de consumo."}

    # Features temporales
    df["mes"]   = pd.to_datetime(df["mes"])
    df["year"]  = df["mes"].dt.year
    df["month"] = df["mes"].dt.month

    X = df[["year", "month", "materia_prima_id"]]
    y = df["consumo"]

    if len(df) < 2:
        return {"mensaje": "Se requieren al menos 2 registros para entrenar el modelo."}

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        os.makedirs("modelos", exist_ok=True)
        with open("modelos/modelo_consumo_mp.pkl", "wb") as f:
            pickle.dump(modelo, f)

        return {
            "mensaje": "El modelo de consumo de materias primas se ha entrenado y guardado correctamente.",
            "r2": round(r2, 3),
            "rmse": round(rmse, 3)
        }

    except Exception as e:
        return {
            "mensaje": "Error al entrenar o guardar el modelo de consumo.",
            "error": str(e)
        }


def predecir_consumo_materia_prima() -> dict:
    """
    Carga el modelo de consumo entrenado y predice la demanda del mes siguiente
    para cada materia prima presente.

    Returns:
        dict: {
            "forecast": [
                {"mes": "YYYY-MM", "materia_prima_id": int, "consumo_pred": float},
                ...
            ]
        }
    """
    modelo_path = "modelos/modelo_consumo_mp.pkl"
    if not os.path.exists(modelo_path):
        return {"forecast": []}

    with open(modelo_path, "rb") as f:
        modelo = pickle.load(f)

    engine = conectar_db()
    # Obtener última fecha de compras registradas
    last_fecha = pd.read_sql(
        "SELECT MAX(fecha) AS last FROM compras", con=engine
    )["last"][0]
    if not last_fecha:
        return {"forecast": []}

    last = pd.to_datetime(last_fecha)
    # calcular primer día del mes siguiente
    next_month = (last + pd.offsets.MonthBegin()).replace(day=1)

    # materiales únicos
    mp_df = pd.read_sql(
        "SELECT DISTINCT materia_prima_id FROM detalle_compra",
        con=engine
    )
    materiales = mp_df["materia_prima_id"].tolist()

    # preparar DataFrame de predicción
    Xf = pd.DataFrame({
        "year": [next_month.year] * len(materiales),
        "month": [next_month.month] * len(materiales),
        "materia_prima_id": materiales
    })

    preds = modelo.predict(Xf)
    forecast = [
        {
            "mes": next_month.strftime("%Y-%m"),
            "materia_prima_id": int(mp),
            "consumo_pred": round(float(p), 2)
        }
        for mp, p in zip(materiales, preds)
    ]

    return {"forecast": forecast}


def obtener_datos_para_entrenamiento() -> list[dict]:
    """
    Trae los datos agregados por cliente para el entrenamiento del modelo.
    Si no hay datos, devuelve una lista vacía.
    """
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

    # Si no hay filas, devolvemos lista vacía en lugar de None
    if not datos:
        return []

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

def entrenar_modelo_regresion_lineal(date_from=None, date_to=None) -> dict:
    """
    Entrena un LinearRegression sobre la variable binaria `volvera_comprar` y guarda el modelo.
    Si no hay datos o ocurre un error, devuelve un mensaje apropiado.
    """
    engine = conectar_db()
    sql, params = _get_info_por_rango(date_from, date_to)
    df = pd.read_sql(sql, con=engine, params=params)

    # Se verifica que haya datos
    if df.empty:
        return {"mensaje": "No hay datos suficientes para entrenar el modelo lineal."}

    # Se crea la variable objetivo, binaria, si(1) o no(0)
    df["volvera_comprar"] = df["dias_desde_ultima_compra"].apply(lambda x: 1 if x < 60 else 0)

    X = df[['edad', 'cantidad_total_pedidos', 'dias_desde_ultima_compra', 'total_gastado']]
    y = df['volvera_comprar']

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

        # Se evalua con métricas de regresión
        y_pred = modelo.predict(X_test)
        r2  = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        # Se guarda el modelo
        os.makedirs("modelos", exist_ok=True)
        with open("modelos/modelo_lineal.pkl", "wb") as f:
            pickle.dump(modelo, f)
        
        # Se devuelve la confirmación y se incluyen métricas
        return {
            "mensaje": "Modelo lineal entrenado y guardado correctamente.",
            "r2": round(r2, 3),
            "rmse": round(rmse, 3)
        }

    except Exception as e:
        return {
            "mensaje": "El modelo lineal no se ha podido entrenar ni guardar correctamente.",
            "error": str(e)
        }

def entrenar_modelo_regresion_logistica(date_from=None, date_to=None) -> dict:
    """
    Entrena un modelo de regresión logística para predecir volvera_comprar (0/1),
    y devuelve el classification_report completo como diccionario.
    """
    engine = conectar_db()
    sql, params = _get_info_por_rango(date_from, date_to)
    df = pd.read_sql(sql, con=engine, params=params)

    if df.empty:
        return {"mensaje": "No hay datos suficientes para entrenar el modelo logístico."}

    # Variable objetivo, binaria, si(1) o no(0)
    df["volvera_comprar"] = df["dias_desde_ultima_compra"].apply(lambda x: 1 if x < 60 else 0)
    X = df[['edad','cantidad_total_pedidos','dias_desde_ultima_compra','total_gastado']]
    y = df['volvera_comprar']

    if len(df) < 2:
        return {"mensaje": "Se requieren al menos 2 registros para entrenar el modelo logístico."}

    try:
        # Se dividen los datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        modelo = LogisticRegression(max_iter=1000)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        # Se obtiene el reporte de clasificación, como diccionario
        report = classification_report(y_test, y_pred, output_dict=True)

        # Se guarda el modelo
        os.makedirs("modelos", exist_ok=True)
        with open("modelos/modelo_logistico.pkl", "wb") as f:
            pickle.dump(modelo, f)

        return {
            "mensaje": "Modelo de regresión logística entrenado y guardado correctamente.",
            "classification_report": report
        }

    except Exception as e:
        return {
            "mensaje": "El modelo de regresión logística no se ha podido entrenar ni guardar correctamente.",
            "error": str(e)
        }

def entrenar_modelo_arbol_decision(
    date_from: Optional[str] = None,
    date_to:   Optional[str] = None
) -> dict:
    """
    Entrena un DecisionTreeClassifier para predecir si un cliente volverá a comprar (0/1),
    a partir de datos de edad, pedidos, días desde última compra y total gastado.
    Valida que haya datos suficientes y devuelve el classification_report.

    Args:
        date_from (str, optional): Fecha mínima 'YYYY-MM-DD' para filtrar el histórico.
        date_to   (str, optional): Fecha máxima 'YYYY-MM-DD' para filtrar el histórico.

    Returns:
        dict: {
            "mensaje": str,
            "classification_report": dict,  # si el entrenamiento fue exitoso
            "error": str                   # solo si ocurre excepción
        }
    """
    engine = conectar_db()
    sql, params = _get_info_por_rango(date_from, date_to)
    df = pd.read_sql(sql, con=engine, params=params)

    if df.empty:
        return {"mensaje": "No hay datos suficientes para entrenar el modelo de árbol de decisión."}

    df["volvera_comprar"] = df["dias_desde_ultima_compra"].apply(lambda x: 1 if x < 60 else 0)

    X = df[['edad', 'cantidad_total_pedidos', 'dias_desde_ultima_compra', 'total_gastado']]
    y = df['volvera_comprar']

    if len(df) < 2:
        return {"mensaje": "Se requieren al menos 2 registros para entrenar y validar el árbol de decisión."}

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        modelo = DecisionTreeClassifier(random_state=42)
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)

        os.makedirs("modelos", exist_ok=True)
        with open("modelos/modelo_arbol.pkl", "wb") as f:
            pickle.dump(modelo, f)

        return {
            "mensaje": "Modelo de árbol de decisión entrenado y guardado correctamente.",
            "classification_report": report
            #"accuracy":  round(acc,  3),
            #"precision": round(prec, 3),
            #"recall":    round(rec,  3),
            #"f1_score":  round(f1,   3)
        }

    except Exception as e:
        return {
            "mensaje": "Error al entrenar o guardar el modelo de árbol de decisión.",
            "error": str(e)
        }


def entrenar_modelo_bosque_aleatorio(
    date_from: Optional[str] = None,
    date_to:   Optional[str] = None
) -> dict:
    """
    Entrena un RandomForestClassifier para predecir si un cliente volverá a comprar (0/1),
    a partir de datos de edad, pedidos, días desde última compra y total gastado.
    Valida que haya datos suficientes y devuelve el classification_report.

    Args:
        date_from (str, optional): Fecha mínima 'YYYY-MM-DD' para filtrar el histórico.
        date_to   (str, optional): Fecha máxima 'YYYY-MM-DD' para filtrar el histórico.

    Returns:
        dict: {
            "mensaje": str,
            "classification_report": dict,  # si el entrenamiento fue exitoso
            "error": str                   # solo si ocurre excepción
        }
    """
    engine = conectar_db()
    sql, params = _get_info_por_rango(date_from, date_to)
    df = pd.read_sql(sql, con=engine, params=params)

    if df.empty:
        return {"mensaje": "No hay datos suficientes para entrenar el modelo de bosque aleatorio."}

    df["volvera_comprar"] = df["dias_desde_ultima_compra"].apply(lambda x: 1 if x < 60 else 0)

    X = df[['edad', 'cantidad_total_pedidos', 'dias_desde_ultima_compra', 'total_gastado']]
    y = df['volvera_comprar']

    if len(df) < 2:
        return {"mensaje": "Se requieren al menos 2 registros para entrenar y validar el bosque aleatorio."}

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)

        os.makedirs("modelos", exist_ok=True)
        with open("modelos/modelo_bosque.pkl", "wb") as f:
            pickle.dump(modelo, f)

        return {
            "mensaje": "Modelo de bosque aleatorio entrenado y guardado correctamente.",
            "classification_report": report
            #"accuracy":  round(acc,  3),
            #"precision": round(prec, 3),
            #"recall":    round(rec,  3),
            #"f1_score":  round(f1,   3)
        }

    except Exception as e:
        return {
            "mensaje": "Error al entrenar o guardar el modelo de bosque aleatorio.",
            "error": str(e)
        }

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

    if df.empty:
        return []
    
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

# Funciones para 
# --------------------------------------------------------------------
def consumo_material_mensual() -> dict:
    """
    Retorna datos para un gráfico de líneas con el consumo mensual de cada materia prima.
    """
    engine = conectar_db()
    sql = """
        SELECT 
          DATE_FORMAT(c.fecha, '%Y-%m') AS mes,
          mp.nombre AS materia_prima_nombre,
          SUM(dc.cantidad) AS total_unidades
        FROM compras c
        JOIN detalle_compra dc 
          ON c.id = dc.compra_id
        JOIN materias_primas mp 
          ON dc.materia_prima_id = mp.id
        GROUP BY mes, mp.nombre
        ORDER BY mes
    """
    df = pd.read_sql(sql, con=engine)

    if df.empty:
        return {"labels": [], "datasets": []}

    pivot = df.pivot_table(
        index="mes", 
        columns="materia_prima_nombre", 
        values="total_unidades",
        aggfunc="sum", # para combinar todas las filas duplicadas sumando sus valores
        fill_value=0 # cuando no hay datos, se rellena con cero
    )

    return {
        # lista de meses ordenados para el eje X
        "labels": pivot.index.tolist(),
        # lista de objetos, uno por cada materia prima
        "datasets": [
            {
                "label": materia, # f"MP {mp_id}" 
                "data": pivot[materia].tolist() # mp_id
            }
            for materia in pivot.columns # mp_id
        ]
    }

def gasto_por_proveedor() -> dict:
    """
    Retorna datos para un gráfico de pastel con la proporción de gasto por proveedor.
    """
    engine = conectar_db()
    sql = """
        SELECT 
          p.razon_social,
          SUM(dc.cantidad * dc.precio_unitario) AS gasto
        FROM compras c
        JOIN detalle_compra dc 
          ON c.id = dc.compra_id
        JOIN proveedores p 
          ON c.proveedor_id = p.id
        GROUP BY p.razon_social
    """
    df = pd.read_sql(sql, con=engine)

    if df.empty:
        return {"labels": [], "data": []}

    return {
        "labels": df["razon_social"].tolist(),
        "data":   df["gasto"].tolist()
    }

def top_materiales() -> dict:
    """
    Retorna datos para un gráfico de barras con los 10 materiales más comprados (unidades).
    """
    engine = conectar_db()
    sql = """
        SELECT 
          mp.nombre,
          SUM(dc.cantidad) AS total_unidades
        FROM detalle_compra dc
        JOIN materias_primas mp 
          ON dc.materia_prima_id = mp.id
        GROUP BY mp.nombre
        ORDER BY total_unidades DESC
        LIMIT 10
    """
    df = pd.read_sql(sql, con=engine)

    if df.empty:
        return {"labels": [], "data": []}

    return {
        "labels": df["nombre"].tolist(),
        "data":   df["total_unidades"].tolist()
    }

def uso_por_color() -> dict:
    """
    Retorna datos para un gráfico de doughnut con el uso porcentual de cada color.
    """
    engine = conectar_db()
    sql = """
        SELECT 
          c.nombre,
          COUNT(*) AS usos
        FROM detalle_compra dc
        JOIN colores c 
          ON dc.color_id = c.id
        GROUP BY c.nombre
    """
    df = pd.read_sql(sql, con=engine)

    if df.empty:
        return {
            "labels": [],
            "data": [],
            "percent": []
        }

    total = df["usos"].sum()
    percent = (df["usos"] / total * 100).round(1)

    return {
        "labels": df["nombre"].tolist(),
        "data":   df["usos"].tolist(),
        "percent": percent.tolist()
    }

def dispersion_precio_cantidad() -> dict:
    """
    Retorna los pares (cantidad, precio_unitario) de todas las líneas de detalle_compra,
    para un gráfico de dispersión precio vs cantidad.

    Returns:
        dict: {
            "data": List[{"cantidad": int, "precio_unitario": float}]
        }
    """
    engine = conectar_db()
    sql = text("SELECT cantidad, precio_unitario FROM detalle_compra")
    df = pd.read_sql(sql, con=engine)

    if df.empty:
        return {"data": []}

    return {"data": df.to_dict(orient="records")}
