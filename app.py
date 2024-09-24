from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Permite solicitudes desde otros dominios, ajusta según seguridad

# Cargar el modelo entrenado
model = joblib.load('modelo_inventario.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Espera datos en formato JSON
    try:
        # Convertir los datos recibidos en un DataFrame
        input_df = pd.DataFrame([data])
        
        # Realizar la predicción
        prediction = model.predict(input_df)
        
        # Devolver el resultado
        return jsonify({'prediccion': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
