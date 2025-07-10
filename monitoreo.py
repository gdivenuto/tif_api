import pandas as pd

from sqlalchemy import text
from typing import Optional
from db import conectar_db

# Funciones para Monitoreo
# --------------------------------------------------------------------
def consumo_material_mensual(
    categoria_id: Optional[int] = None
) -> dict:
    """
    Retorna datos para un gráfico de líneas con el consumo mensual de cada materia prima.
    Si `categoria_id` se pasa, filtra por esa categoría; si no, incluye todas las categorías.
    """
    engine = conectar_db()
    filtros = []
    params = {}

    if categoria_id is not None:
        filtros.append("mp.categoria_id = :cat_id")
        params["cat_id"] = categoria_id

    where = f"WHERE {' AND '.join(filtros)}" if filtros else ""

    sql = f"""
        SELECT 
          DATE_FORMAT(c.fecha, '%Y-%m') AS mes,
          mp.nombre AS materia_prima_nombre,
          SUM(dc.cantidad) AS total_unidades
        FROM compras c
        JOIN detalle_compra dc 
          ON c.id = dc.compra_id
        JOIN materias_primas mp 
          ON dc.materia_prima_id = mp.id
        {where}
        GROUP BY mes, mp.nombre
        ORDER BY mes
    """

    df = pd.read_sql(text(sql), con=engine, params=params)

    if df.empty:
        return {"labels": [], "datasets": []}

    pivot = df.pivot_table(
        index="mes",
        columns="materia_prima_nombre",
        values="total_unidades",
        aggfunc="sum",
        fill_value=0
    )

    return {
        "labels": pivot.index.tolist(),
        "datasets": [
            {"label": materia, "data": pivot[materia].tolist()}
            for materia in pivot.columns
        ]
    }

def consumo_material_mensual_old() -> dict:
    """
    Retorna datos para un gráfico de líneas con el consumo mensual de cada materia prima.
    """
    engine = conectar_db()
    sql = """
        SELECT 
          DATE_FORMAT(c.fecha, '%%Y-%%m') AS mes,
          mp.nombre AS materia_prima_nombre,
          SUM(dc.cantidad) AS total_unidades
        FROM compras c
        JOIN detalle_compra dc 
          ON c.id = dc.compra_id
        JOIN materias_primas mp 
          ON dc.materia_prima_id = mp.id
        WHERE mp.categoria_id = 7
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
