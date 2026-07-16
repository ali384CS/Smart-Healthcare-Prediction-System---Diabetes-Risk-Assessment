from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Get the absolute path of the directory containing app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models using the absolute path
model = joblib.load(os.path.join(BASE_DIR, "diabetes_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # Extract 10 features directly as floats
        features = np.array([[
            float(data["age"]),
            float(data["time_in_hospital"]),
            float(data["num_lab_procedures"]),
            float(data["num_procedures"]),
            float(data["num_medications"]),
            float(data["number_diagnoses"]),
            float(data["max_glu_serum"]),
            float(data["A1Cresult"]),
            float(data["change"]),
            float(data["diabetesMed"])
        ]])
        
        features_scaled = scaler.transform(features)
        res = int(model.predict(features_scaled)[0])
        
        mapping = {
            0: ("Non-Diabetic", "Recommendation: Maintain a balanced diet and regular exercise. Your risk indicators are normal."),
            1: ("Pre-Diabetic", "Recommendation: Adopt a low-sugar diet and increase physical activity. Monitor your A1C and Glucose levels closely."),
            2: ("Diabetic", "Recommendation: Immediate clinical consultation required. Continue prescribed medications and regular glucose monitoring.")
        }
        
        status, advice = mapping[res]
        return jsonify({"prediction": status, "recommendation": advice})
        
    except Exception as e:
        print("Error: " + str(e))
        return jsonify({"prediction": "Error", "recommendation": "Check backend connection or input formats."}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)