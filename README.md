
# API para la creación y uso de Modelos de Aprendizaje Supervisado para el TIF

### Creación del directorio
mkdir tif_api

cd tif_api

### Si no se encuentra instalado Python aún
#### Actualizar primero la lista de paquetes disponibles en los repositorios del sistema
sudo apt update
#### Instalar Python
sudo apt install python3
#### Verificar si se ha instalado y su versión
python -V

python --version

### Creación del entorno virtual
python3 -m venv venv

Esto creará una carpeta venv/ dentro del proyecto con una instalación aislada de Python.

La bandera -m ejecuta el módulo de Python como un script.

### Activar el entorno virtual
source venv/bin/activate

Se verá en la terminal: (venv) (base) thor@thor-desktop:/var/www/html/tif_api$

### Instalar las dependencias necesarias
Es recomendable contar con un archivo de texto, requirements.txt, que contenga el nombre de todas las dependencias que el proyecto requiere. las cuales son: fastapi, uvicorn, scikit-learn, pandas, mysql-connector-python, joblib

Para instalarlas se debe ejecutar:

pip install -r requirements.txt

pip es un sistema de gestión de paquetes, Python 3 ya lo incluye.

La bandera -r permite leer e instalar paquetes desde el archivo requirements.txt

### Iniciar el servidor para verificar la instalación
uvicorn main:app --reload

### Crear el directorio donde se guardarán los modelos entrenados
mkdir modelos

### Resultado de la estructura del proyecto, con los diferentes archivos necesarios para el proyecto del Modelo de AAS del TIF:
tif_api/

├── venv/                  # entorno virtual

├── main.py                # API con diversos endpoints para la utilización de los modelos, para su desarrollo se utilizó FastAPI

├── db.py                  # Método que retorna la conexión a la base de datos MySQL del TIF

├── model_utils.py         # Métodos para crear, entrenar y usar diferentes modelos, para realizar predicciones.

├── modelos/

--└── modelo.pkl                     # Archivo que contiene el modelo guardado de Regresión Lineal

--└── modelo_logistica.pkl           # Archivo que contiene el modelo guardado de Regresión Logística

--└── modelo_con_arbol.pkl           # Archivo que contiene el modelo guardado de Arbol de Decisión

--└── modelo_con_bosque.pkl          # Archivo que contiene el modelo guardado de Bosques Aleatorios

└── requirements.txt       # dependencias del proyecto para el modelo

## Importante, al iniciar la API
### Primero se debe activar el entorno virtual
source venv/bin/activate

Se verá en la terminal: (venv) (base) thor@thor-desktop:/var/www/html/tif_api$

### Luego iniciar el servidor, se utiliza --reload para que tome los cambios en la API
uvicorn main:app --reload

### Acceso a la documentación interactiva generada por FastAPI:
http://127.0.0.1:8000/docs     ← Swagger UI

http://127.0.0.1:8000/redoc    ← ReDoc
