
# API para la creación y uso de Modelos de Aprendizaje Supervisado para el TIF

### Instalación de Python
#### Actualizar primero la lista de paquetes disponibles en los repositorios del sistema
```bash
sudo apt update
```
#### Instalar la versión 3 de Python
```bash
sudo apt install python3
```
#### Verificar si se ha instalado y su versión
```bash
python3 --version
```
### Creación del directorio de la API
El directorio de la API del proyecto TIF deberá ubicarse en /var/www/html/tif_api/
```bash
cd /var/www/html/
mkdir tif_api
cd tif_api
```
### Creación del entorno virtual
```bash
sudo apt install python3.12-venv

python3 -m venv venv
```
Esto creará una carpeta venv/ dentro del proyecto con una instalación aislada de Python.

La bandera -m ejecuta el módulo de Python como un script.

### Activar el entorno virtual
```bash
source venv/bin/activate
```
Se verá en la terminal: (venv) (base) thor@thor-desktop:/var/www/html/tif_api$

### Instalar las dependencias necesarias
Es recomendable contar con un archivo de texto, requirements.txt, que contenga el nombre de todas las dependencias que el proyecto requiere. las cuales son: fastapi, uvicorn, scikit-learn, pandas, mysql-connector-python, pymysql, joblib, sqlalchemy, cryptography

Para instalarlas se debe ejecutar:
```bash
pip install -r requirements.txt
```
pip es un sistema de gestión de paquetes, Python 3 ya lo incluye.

La bandera -r permite leer e instalar paquetes desde el archivo requirements.txt

### Crear el directorio donde se guardarán los modelos entrenados
```bash
mkdir modelos
```
### Estructura del proyecto de la API
Con los archivos necesarios para la creación, entrenamiento y uso de Modelos de Aprendizaje Automático Supervisado.

```bash
tif_api/
├── modelos/
│   ├── modelo_consumo_mp.pkl       # Regresión con Bosques Aleatorios.
│   ├── modelo_demanda.pkl          # Regresión con Bosques Aleatorios.
│   ├── modelo_lineal_ventas.pkl    # Modelo de Regresión Lineal Múltiple (3 variables independientes).
│   └── modelo_lineal_ventas_v2.pkl # Modelo de Regresión Lineal Múltiple (2 variables independientes).
├── venv/                           # Directorio del entorno virtual.
├── db.py                           # Se obtiene un pool de conexiones a la base de datos.
├── entrenar_modelo_consumo_mp.py   # Se crea y entrena el modelo de compras.
├── entrenar_modelos_venta_v2.py    # Se crea y entrena el modelo de ventas.
├── entrenar_modelos_venta.py       # Se crea y entrena el modelo de ventas.
├── main.py                         # Se define la API para la utilización de los modelos.
├── monitoreo.py                    # Se obtienen datos para las gráficas de Monitoreo.
├── predecir_consumo_mp.py          # Se utiliza el modelo de compras para predecir la demanda de materias primas.
├── predecir_demanda_mensual_mp.py  # Se utiliza el modelo de compras para predecir la demanda mensual de materias primas.
├── predicciones_ventas.py          # Se utiliza el modelo de ventas para realizar diversas predicciones.
└── requirements.txt                # Contiene las dependencias del proyecto para la API.
```

## Iniciar la API para su uso
### Activar el entorno virtual
```bash
source venv/bin/activate
```
Se verá en la terminal: (venv) (base) thor@thor-desktop:/var/www/html/tif_api$

### Iniciar el servidor de la API
```bash
uvicorn main:app --reload
```
(Se utiliza --reload para que tome los cambios en la API)

### Acceso a la documentación interactiva generada por FastAPI:
Swagger UI: http://127.0.0.1:8000/docs

ReDoc: http://127.0.0.1:8000/redoc
