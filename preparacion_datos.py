import pandas as pd
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Configuración de la conexión a la base de datos
config = {
    'user': 'tu_usuario',
    'password': 'tu_contraseña',
    'host': 'localhost',
    'database': 'tu_base_de_datos',
    'raise_on_warnings': True
}

def extract_data():
    # Conectar a la base de datos
    cnx = mysql.connector.connect(**config)
    query = "SELECT * FROM inventario"  # Ajusta la consulta según tus necesidades
    df = pd.read_sql(query, cnx)
    cnx.close()
    return df

def prepare_data(df):
    # Ejemplo simple de preprocesamiento
    df = df.dropna()  # Eliminar filas con valores nulos
    # Codificar variables categóricas si las hay
    # df = pd.get_dummies(df, columns=['categorica'])
    return df

def train_model(df):
    # Definir características (X) y variable objetivo (y)
    X = df.drop('demanda_futura', axis=1)  # Ajusta 'demanda_futura' según tu caso
    y = df['demanda_futura']
    
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Error Cuadrático Medio: {mse}")
    
    # Guardar el modelo entrenado
    joblib.dump(model, 'modelo_inventario.pkl')
    print("Modelo guardado como 'modelo_inventario.pkl'")

def main():
    print("Extrayendo datos...")
    df = extract_data()
    print("Preparando datos...")
    df = prepare_data(df)
    print("Entrenando modelo...")
    train_model(df)
    print("Proceso completado.")

if __name__ == "__main__":
    main()
