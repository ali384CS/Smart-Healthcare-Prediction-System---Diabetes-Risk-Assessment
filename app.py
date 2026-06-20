from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

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