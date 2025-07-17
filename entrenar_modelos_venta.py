import pickle
import os
import pandas as pd
from typing import Optional

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    root_mean_squared_error, # Pérdida de regresión del error cuadrático medio
    mean_squared_error, # Raíz del error cuadrático medio (OBSOLETO)
    r2_score, # Coeficiente de determinación: función de puntuación de regresión R^2
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    explained_variance_score,
    classification_report, # reporte con las principales métricas de clasificación
    accuracy_score, # proporción de aciertos global
    precision_score, # de los positivos predichos, cuántos son correctos
    recall_score, # sensibilidad de los positivos reales, cuántos se aciertan
    f1_score, # media armónica de precision y recall
)
from sqlalchemy import text
from fastapi import HTTPException

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
    # c.edad,
    # , c.edad
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
    GROUP BY c.id
    """
    return sql, tuple(params)

def entrenar_modelo_venta_regresion_lineal(
    date_from: Optional[str] = None,
    date_to:   Optional[str] = None
) -> dict:
    """
    Entrena un LinearRegression para estimar el total gastado de cada cliente (variable continua) 
    A partir de:
      - edad
      - cantidad_total_pedidos
      - dias_desde_ultima_compra

    Args:
        date_from (str, optional): fecha mínima 'YYYY-MM-DD' para filtrar histórico.
        date_to   (str, optional): fecha máxima 'YYYY-MM-DD' para filtrar histórico.

    Returns:
        dict: {
          "mensaje": str,
          "r2": float,    # Coeficiente de determinación
          "rmse": float,  # Raíz del Error Cuadrático Medio
          "mae": float,   # Error Medio Absoluto
          "mape": float,  # Error Porcentual Absoluto Medio
          "medae": float, # Mediana del Error Absoluto
        }
    """
    engine = conectar_db()
    sql, params = _get_info_ventas(date_from, date_to)
    df = pd.read_sql(sql, con=engine, params=params)

    # Se verifica que haya datos
    if df.empty:
        return {"mensaje": "No hay datos suficientes para entrenar el modelo lineal de ventas."}

    # Se crea la variable objetivo, binaria, si(1) o no(0)
    df["volvera_comprar"] = df["dias_desde_ultima_compra"].apply(lambda x: 1 if x < 60 else 0)

    # Features
    X = df[['cantidad_total_pedidos', 'dias_desde_ultima_compra']]#'edad', 
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

        # Se obtienen métricas
        y_pred = modelo.predict(X_test)
        r2    = r2_score(y_test, y_pred)
        rmse  = root_mean_squared_error(y_test, y_pred)
        mae   = mean_absolute_error(y_test, y_pred)
        mape  = mean_absolute_percentage_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        #ev   = explained_variance_score(y_test, y_pred)

        # Se guarda el modelo
        os.makedirs("modelos", exist_ok=True)
        with open("modelos/modelo_lineal_ventas.pkl", "wb") as f:
            pickle.dump(modelo, f)
        
        # Se devuelve la confirmación y se incluyen métricas
        return {
            "mensaje": "Modelo lineal de ventas entrenado y guardado correctamente.",
            "r2": round(r2, 3),
            "rmse": round(rmse, 3),
            "mae":   round(mae,   3),
            "mape":  round(mape,  3),
            "medae": round(medae, 3),
            #"explained_variance": round(ev, 3)
        }

    except Exception as e:
        return {
            "mensaje": "Error al entrenar o guardar el modelo de ventas.",
            "error": str(e)
        }

def entrenar_modelo_regresion_logistica(date_from=None, date_to=None) -> dict:
    """
    Entrena un modelo de regresión logística para predecir volvera_comprar (0/1),
    y devuelve el classification_report completo como diccionario.
    """
    engine = conectar_db()
    sql, params = _get_info_ventas(date_from, date_to)
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
    sql, params = _get_info_ventas(date_from, date_to)
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
    sql, params = _get_info_ventas(date_from, date_to)
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
