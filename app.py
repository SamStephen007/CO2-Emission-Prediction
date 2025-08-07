from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('model/co2_model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form.get(feat)) for feat in ['engine_size', 'cylinders', 'city', 'highway', 'combined', 'mpg']]
        scaled_data = scaler.transform([data])
        prediction = model.predict(scaled_data)[0]
        return render_template('index.html', prediction_text=f'Estimated CO₂ Emission: {round(prediction, 2)} g/km')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
