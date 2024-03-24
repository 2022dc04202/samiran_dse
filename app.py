from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('train_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    features = request.json['features']
    input_data_as_numpy_array = np.asarray(features)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run()