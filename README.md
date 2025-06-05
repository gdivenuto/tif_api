# tif_api
API para el uso de un Modelo de Aprendizaje Supervisado

## Pasos para crear un entorno virtual para el proyecto del Modelo de AAS del TIF
### Creación del directorio
mkdir tif_api

cd tif_api

### Creación del entorno virtual
python3 -m venv venv

Esto creará una carpeta venv/ dentro del proyecto con una instalación aislada de Python.

### Activar el entorno virtual
source venv/bin/activate

Se verá en la terminal: (venv) (base) thor@thor-desktop:/var/www/html/tif_api$

### Instalar las dependencias necesarias
Es recomendable contar con un archivo de texto, requirements.txt, que contenga el nombre de todas las dependencias que el proyecto requiere. las cuales son: fastapi, uvicorn, scikit-learn, pandas, mysql-connector-python, joblib

Para instalarlas se debe ejecutar:

pip install -r requirements.txt

### Verificar la instalación
uvicorn main:app --reload

### Resultado de la estructura del proyecto, con los diferentes archivos necesarios para el proyecto del Modelo de AAS del TIF:
tif_api/

├── venv/                  # entorno virtual

├── main.py                # API con diversos endpoints para la utilización del modelo, para su desarrollo se utilizó FastAPI

├── db.py                  # Método que retorna la conexión a la base de datos MySQL del TIF

├── model_utils.py         # Métodos para el entrenamiento y diferentes predicciones con dicho modelo creado

├── modelos/

--└── modelo.pkl           # Archivo que contiene el modelo guardado

└── requirements.txt       # dependencias del proyecto para el modelo

### Acceso a la documentación interactiva generada por FastAPI:
http://127.0.0.1:8000/docs     ← Swagger UI

http://127.0.0.1:8000/redoc    ← ReDoc
