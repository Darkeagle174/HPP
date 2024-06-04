from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    beds = int(data['beds'])
    baths = float(data['baths'])
    size = float(data['size'])
    lot_size = float(data['lot_size'])
    zip_code = int(data['zip_code'])

    user_input = pd.DataFrame({
        'beds': [beds],
        'baths': [baths],
        'size': [size],
        'lot_size': [lot_size],
        'zip_code': [zip_code]
    })

    user_input_scaled = scaler.transform(user_input)
    price_pred = model.predict(user_input_scaled)

    return jsonify({'price': price_pred[0]})

if __name__ == '__main__':
    app.run(debug=True)
