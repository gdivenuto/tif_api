from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model_utils import predecir_con_modelo, entrenar_y_guardar_modelo, predecir_por_cliente_id, debug_estructura_detalle_pedido
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

@app.get("/debug_table_detalle_pedido")
def debug_table_detalle_pedido():
    return debug_estructura_detalle_pedido()

@app.get("/entrenar")
def entrenar():
    entrenar_y_guardar_modelo()
    return {"mensaje": "Modelo entrenado y guardado"}

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
