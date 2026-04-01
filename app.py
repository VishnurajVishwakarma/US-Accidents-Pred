from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os
import json
import datetime

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)  # Enable CORS for frontend

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and preprocessors
MODEL_PATH = os.path.join(BASE_DIR, "models/accident_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models/label_encoders.pkl")
FEAT_COLS_PATH = os.path.join(BASE_DIR, "models/feature_columns.pkl")

import tempfile

if not os.path.exists(MODEL_PATH):
    # Try reconstructing from chunks if they exist
    part_files = sorted([f for f in os.listdir(os.path.join(BASE_DIR, "models")) if f.startswith("part_")])
    if part_files:
        print("Reconstructing model from chunks...")
        tmp_model_path = os.path.join(tempfile.gettempdir(), "accident_model.pkl")
        if not os.path.exists(tmp_model_path):
            with open(tmp_model_path, "wb") as outfile:
                for part in part_files:
                    with open(os.path.join(BASE_DIR, "models", part), "rb") as infile:
                        outfile.write(infile.read())
        MODEL_PATH = tmp_model_path

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le_dict = joblib.load(ENCODER_PATH)
    feature_columns = joblib.load(FEAT_COLS_PATH)
else:
    model, scaler, le_dict, feature_columns = None, None, None, None
    print("Warning: Model files not found. Run training script first.")

@app.route("/")
def serve_index():
    return send_from_directory('frontend', 'index.html')

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory('frontend', path)

@app.route("/data/heatmap")
def serve_heatmap():
    heatmap_path = os.path.join(BASE_DIR, "data/heatmap_sample.json")
    if os.path.exists(heatmap_path):
        with open(heatmap_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify([])

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
        
    data = request.json
    
    if isinstance(data, list):
        df_input = pd.DataFrame(data)
    else:
        df_input = pd.DataFrame([data])
        
    # We need to construct a dataframe that matches feature_columns exactly.
    df = pd.DataFrame(index=df_input.index, columns=feature_columns)
    
    # Fill with provided data or sensible defaults
    now = datetime.datetime.now()
    
    for col in feature_columns:
        if col in df_input.columns:
            df[col] = df_input[col]
        else:
            # Provide default values based on column type/name
            if col == 'year': df[col] = now.year
            elif col == 'month': df[col] = now.month
            elif col == 'day': df[col] = now.day
            elif col == 'hour': df[col] = now.hour
            elif col == 'Temperature(F)': df[col] = 65.0
            elif col == 'Humidity(%)': df[col] = 50.0
            elif col == 'Pressure(in)': df[col] = 30.0
            elif col == 'Visibility(mi)': df[col] = 10.0
            elif col == 'Wind_Speed(mph)': df[col] = 5.0
            elif col == 'duration': df[col] = 3600 # 1 hour default
            else:
                # Use default category or 0
                if col in le_dict:
                    df[col] = le_dict[col].classes_[0]
                else:
                    df[col] = 0
                
    # Encode categorical
    for col, le in le_dict.items():
        if col in df.columns:
            # Handle unseen labels by falling back to the first class
            df[col] = df[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
            
    # Scale numerical
    if hasattr(scaler, 'feature_names_in_'):
        num_cols = scaler.feature_names_in_
        df[num_cols] = scaler.transform(df[num_cols])
    else:
        df[:] = scaler.transform(df)
        
    try:
        # Generate probabilities to create a more precise continuous score.
        probabilities = model.predict_proba(df)
        classes = model.classes_
        
        expected_values = []
        for prob in probabilities:
            # Expected value = sum of (probability * class_value)
            ev = sum(p * float(c) for p, c in zip(prob, classes))
            expected_values.append(ev)
            
        if isinstance(data, list):
            return jsonify({"severities": expected_values})
        else:
            return jsonify({"severity": expected_values[0]})
    except Exception as e:
        # Fallback to discrete class prediction if predict_proba is unsupported
        predictions = model.predict(df)
        if isinstance(data, list):
            return jsonify({"severities": [float(p) for p in predictions]})
        else:
            return jsonify({"severity": float(predictions[0])})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
