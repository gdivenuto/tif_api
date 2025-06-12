from sqlalchemy import create_engine

# Parámetros de conexión
HOST     = "localhost"
USER     = "root"
PASSWORD = "gabilan"
DBNAME   = "uaa_tif"

# URL de SQLAlchemy
DATABASE_URL = f"mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}/{DBNAME}"

# Se crea el engine(pool de conexiones) una sola vez al importar este módulo
engine = create_engine(DATABASE_URL, echo=False, future=True)

def conectar_db():
    """
    Devuelve el engine de SQLAlchemy.
    Para operaciones con pandas.read_sql() o con engine.connect()
    """
    return engine
