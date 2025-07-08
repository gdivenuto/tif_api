from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Parámetros de conexión
HOST     = "localhost"
USER     = "root"
PASSWORD = "gabilan"
DBNAME   = "uaa_tif"

# URL de SQLAlchemy
DATABASE_URL = f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}/{DBNAME}"

# Se crea el engine(pool de conexiones) una sola vez al importar este módulo
engine = create_engine(DATABASE_URL, echo=False, future=True, pool_pre_ping=True)

def conectar_db() -> Engine:
    """
    Crea y devuelve un SQLAlchemy Engine (pool de conexiones) para la base de datos MySQL.

    Returns:
        sqlalchemy.Engine: engine con conexión pool a la base de datos.
    """
    return engine
