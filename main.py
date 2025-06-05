from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model_utils import entrenar_y_guardar_modelo, entrenar_logistica, entrenar_modelo_arbol, entrenar_modelo_bosque
from model_utils import predecir_con_modelo, predecir_por_cliente_id, predecir_logistico, predecir_con_arbol, predecir_con_bosque
from pydantic import BaseModel

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

@app.get("/entrenar")
def entrenar():
    entrenar_y_guardar_modelo()
    return {"mensaje": "Modelo entrenado y guardado"}

@app.get("/entrenar_logistica")
def entrenar_logistica_endpoint():
    return entrenar_logistica()

@app.get("/entrenar_arbol")
def entrenar_arbol():
    return entrenar_modelo_arbol()

@app.get("/entrenar_bosque")
def entrenar_bosque():
    return entrenar_modelo_bosque()

@app.post("/predecir")
def predecir(datos: DatosEntrada):
    resultado = predecir_con_modelo(
        datos.edad,
        datos.cantidad_total_pedidos,
        datos.dias_desde_ultima_compra,
        datos.total_gastado
    )
    return {"resultado": round(resultado, 2)}

@app.get("/predecir_por_cliente/{cliente_id}")
def predecir_por_cliente(cliente_id: int):
    return predecir_por_cliente_id(cliente_id)

@app.post("/predecir_logistico")
def predecir_logistico_endpoint(datos: DatosEntrada):
    return predecir_logistico(
        datos.edad,
        datos.cantidad_total_pedidos,
        datos.dias_desde_ultima_compra,
        datos.total_gastado
    )

@app.post("/predecir_con_arbol")
def predecir_con_arbol(datos: DatosEntrada):
    return predecir_con_arbol(
        datos.edad,
        datos.cantidad_total_pedidos,
        datos.dias_desde_ultima_compra,
        datos.total_gastado
    )

@app.post("/predecir_con_bosque")
def predecir_con_bosque(datos: DatosEntrada):
    return predecir_con_bosque(
        datos.edad,
        datos.cantidad_total_pedidos,
        datos.dias_desde_ultima_compra,
        datos.total_gastado
    )