from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model_utils import entrenar_modelo_regresion_lineal, entrenar_modelo_regresion_logistica, entrenar_modelo_arbol_decision, entrenar_modelo_bosque_aleatorio
from model_utils import predecir_por_cliente_id, predecir_con_modelo_lineal, predecir_con_modelo_logistico, predecir_con_modelo_arbol, predecir_con_modelo_bosque
from model_utils import obtener_clientes_para_proyeccion, forecast_demanda_mensual
from pydantic import BaseModel
from typing import Optional

class RangoEntrenamiento(BaseModel):
    date_from: Optional[str]  # ISO 'YYYY-MM-DD'
    date_to:   Optional[str]

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

@app.get("/")
def inicio():
    return {"mensaje": "API de predicci\u00f3n activa"}

@app.post("/entrenar_lineal")
def entrenar_lineal(rango: RangoEntrenamiento):
    return entrenar_modelo_regresion_lineal(rango.date_from, rango.date_to)

@app.post("/entrenar_logistico")
def entrenar_logistico(rango: RangoEntrenamiento):
    return entrenar_modelo_regresion_logistica(rango.date_from, rango.date_to)

@app.post("/entrenar_arbol")
def entrenar_arbol(rango: RangoEntrenamiento):
    return entrenar_modelo_arbol_decision(rango.date_from, rango.date_to)

@app.post("/entrenar_bosque")
def entrenar_bosque(rango: RangoEntrenamiento):
    return entrenar_modelo_bosque_aleatorio(rango.date_from, rango.date_to)

@app.get("/predecir_por_cliente/{cliente_id}")
def predecir_por_cliente(cliente_id: int):
    return predecir_por_cliente_id(cliente_id)

@app.post("/predecir_lineal")
def predecir_lineal(datos: DatosEntrada):
    return predecir_con_modelo_lineal(
        datos.edad,
        datos.cantidad_total_pedidos,
        datos.dias_desde_ultima_compra,
        datos.total_gastado
    )

@app.post("/predecir_logistico")
def predecir_logistico(datos: DatosEntrada):
    return predecir_con_modelo_logistico(
        datos.edad,
        datos.cantidad_total_pedidos,
        datos.dias_desde_ultima_compra,
        datos.total_gastado
    )

@app.post("/predecir_con_arbol")
def predecir_con_arbol(datos: DatosEntrada):
    return predecir_con_modelo_arbol(
        datos.edad,
        datos.cantidad_total_pedidos,
        datos.dias_desde_ultima_compra,
        datos.total_gastado
    )

@app.post("/predecir_con_bosque")
def predecir_con_bosque(datos: DatosEntrada):
    return predecir_con_modelo_bosque(
        datos.edad,
        datos.cantidad_total_pedidos,
        datos.dias_desde_ultima_compra,
        datos.total_gastado
    )

@app.get("/clientes_para_proyeccion")
def clientes_para_proyeccion():
    return obtener_clientes_para_proyeccion()

@app.get("/forecast_demanda")
def forecast_demanda(periodos: int = 12, date_from: str = None, date_to: str = None):
    return forecast_demanda_mensual(periodos, date_from, date_to)
