from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

# =========================================================
# Flask App
# =========================================================
app = Flask(__name__)
CORS(app)

# =========================================================
# Load Model Once
# =========================================================
model = joblib.load("house_price_model.pkl")

# =========================================================
# Home Route
# =========================================================
@app.route('/')
def home():
    return jsonify({
        "message": "House Price Prediction API Running"
    })

# =========================================================
# Prediction Route
# =========================================================
@app.route('/predict', methods=['POST'])
def predict():

    try:
        data = request.get_json()

        # Fast NumPy array
        features = np.array([[
            float(data['income']),
            float(data['house_age']),
            float(data['rooms']),
            float(data['bedrooms']),
            float(data['population'])
        ]], dtype=np.float32)

        # Prediction
        prediction = model.predict(features)[0]

        return jsonify({
            "success": True,
            "predicted_price": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

# =========================================================
# Run App
# =========================================================
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )