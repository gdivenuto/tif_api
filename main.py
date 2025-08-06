from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Entrenamiento y predicción de Compras
# --------------------------------------------------------------
from entrenar_modelo_consumo_mp import entrenar_modelo_consumo_materia_prima
from predecir_consumo_mp import predecir_consumo_materia_prima
from predecir_demanda_mensual_mp import forecast_demanda_mensual

# Entrenamiento y predicción de Ventas
# --------------------------------------------------------------
# from entrenar_modelos_venta import (
#     entrenar_modelo_venta_regresion_lineal, 
#     entrenar_modelo_regresion_logistica, 
#     entrenar_modelo_arbol_decision, 
#     entrenar_modelo_bosque_aleatorio,
# )
from entrenar_modelo_venta_v2 import entrenar_modelo_venta_regresion_lineal

from predicciones_ventas import (
    obtener_clientes_que_compraron, 
    predecir_ventas_futuras_por_cliente,
    predecir_ventas_futuras,
    predecir_ventas_con_modelo_lineal, 
    predecir_ventas_con_modelo_logistico, 
    predecir_ventas_con_modelo_arbol, 
    predecir_ventas_con_modelo_bosque,
)

# Para las Gráficas de Monitoreo
# --------------------------------------------------------------
from monitoreo import (
    consumo_material_mensual,
    gasto_por_proveedor,
    top_materiales,
    uso_por_color,
    dispersion_precio_cantidad,
)

class RangoEntrenamiento(BaseModel):
    date_from: Optional[str] = None # ISO 'YYYY-MM-DD'
    date_to:   Optional[str] = None

class DatosEntrada(BaseModel):
    edad: int
    cantidad_total_pedidos: int
    dias_desde_ultima_compra: int
    total_gastado: float

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Endpoints ------------------------------------------------------
# ----------------------------------------------------------------
# Para probar si la API se encuentra activa
@app.get("/")
def inicio():
    return {"mensaje": "API de modelos de AAS activa"}

# Para el modelo de Consumo en Unidades de Materias Primas ---------------------
@app.post("/entrenar_consumo_mp")
def entrenar_consumo_mp(rango: RangoEntrenamiento):
    return entrenar_modelo_consumo_materia_prima(rango.date_from, rango.date_to)

@app.get("/predecir_consumo_mp")
def predecir_consumo_mp():
    return predecir_consumo_materia_prima()

# Para el Pronóstico de Demanda mensual de Materias Primas
@app.get("/forecast_demanda_mp")
def forecast_demanda_mp(periodos: int = 12, date_from: str = None, date_to: str = None):
    return forecast_demanda_mensual(periodos, date_from, date_to)

# Para entrenamientos de modelos de venta -----------------------------------------------
@app.post("/entrenar_ventas_lineal")
def entrenar_ventas_lineal(rango: RangoEntrenamiento):
    return entrenar_modelo_venta_regresion_lineal(rango.date_from, rango.date_to)

# @app.post("/entrenar_logistico")
# def entrenar_logistico(rango: RangoEntrenamiento):
#     return entrenar_modelo_regresion_logistica(rango.date_from, rango.date_to)

# @app.post("/entrenar_arbol")
# def entrenar_arbol(rango: RangoEntrenamiento):
#     return entrenar_modelo_arbol_decision(rango.date_from, rango.date_to)

# @app.post("/entrenar_bosque")
# def entrenar_bosque(rango: RangoEntrenamiento):
#     return entrenar_modelo_bosque_aleatorio(rango.date_from, rango.date_to)

# Para predecir por un Cliente especifico ------------------------------------------------
@app.get("/predecir_ventas_por_cliente/{cliente_id}")
def predecir_ventas_por_cliente(cliente_id: int):
    result = predecir_ventas_futuras_por_cliente(cliente_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

# Para todos los clientes
@app.get("/predecir_ventas")
def predecir_ventas():
    return predecir_ventas_futuras()

# Para predicciones en Ventas ------------------------------------------------------------
@app.post("/predecir_ventas_lineal")
def predecir_ventas_lineal(datos: DatosEntrada):
    return predecir_ventas_con_modelo_lineal(
        datos.edad,
        datos.cantidad_total_pedidos,
        datos.dias_desde_ultima_compra,
        datos.total_gastado
    )

@app.post("/predecir_logistico")
def predecir_logistico(datos: DatosEntrada):
    return predecir_ventas_con_modelo_logistico(
        datos.edad,
        datos.cantidad_total_pedidos,
        datos.dias_desde_ultima_compra,
        datos.total_gastado
    )

@app.post("/predecir_con_arbol")
def predecir_con_arbol(datos: DatosEntrada):
    return predecir_ventas_con_modelo_arbol(
        datos.edad,
        datos.cantidad_total_pedidos,
        datos.dias_desde_ultima_compra,
        datos.total_gastado
    )

@app.post("/predecir_con_bosque")
def predecir_con_bosque(datos: DatosEntrada):
    return predecir_ventas_con_modelo_bosque(
        datos.edad,
        datos.cantidad_total_pedidos,
        datos.dias_desde_ultima_compra,
        datos.total_gastado
    )
# -----------------------------------------------------------------
# Devuelve los Clientes que compraron, con su monto total gastado
@app.get("/clientes_que_compraron")
def clientes_que_compraron():
    return obtener_clientes_que_compraron()

# Para las Gráficas de Monitoreo -----------------------------------
@app.get("/chart/consumo_material_mensual")
def chart_consumo_material_mensual(
    categoria_id: Optional[int] = Query(
        None,
        title="ID de categoría",
        description="Si se utiliza, filtra la materia prima por dicha categoría."
    )
):
    return consumo_material_mensual(categoria_id)

@app.get("/chart/gasto_por_proveedor")
def chart_gasto_por_proveedor():
    return gasto_por_proveedor()

@app.get("/chart/top_materiales")
def chart_top_materiales():
    return top_materiales()

@app.get("/chart/uso_por_color")
def chart_uso_por_color():
    return uso_por_color()

@app.get("/chart/dispersion_precio_cantidad")
def chart_dispersion_precio_cantidad_endpoint():
    return dispersion_precio_cantidad()