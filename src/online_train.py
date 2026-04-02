import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')
import kagglehub
from kagglehub import KaggleDatasetAdapter

MODEL_PATH = "models/accident_model.pkl"
PROGRESS_PATH = "models/progress.json"

def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, 'r') as f:
            return json.load(f)
    return {"last_chunk": 0, "best_accuracy": 0.0}

def save_progress(progress):
    with open(PROGRESS_PATH, 'w') as f:
        json.dump(progress, f)

def fetch_data_chunk(progress):
    print("🔄 Fetching Kaggle Dataset...")
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "sobhanmoosavi/us-accidents",
        "US_Accidents_March23.csv"
    )
    
    total_chunks = 10
    total_rows = len(df)
    chunk_size = total_rows // total_chunks
    
    start_idx = progress["last_chunk"] * chunk_size
    end_idx = start_idx + chunk_size
    chunk = df.iloc[start_idx:end_idx]
    
    print(f"✅ Loaded sequential chunk {progress['last_chunk']+1}/{total_chunks}. Shape: {chunk.shape}")
    
    # Increment progress safely
    progress["last_chunk"] += 1
    if progress["last_chunk"] >= total_chunks:
        progress["last_chunk"] = 0 # Loop back once consumed
        
    return chunk

def process_and_train(df, progress):
    print("⚙️ Processing chunk...")
    drop_cols = ["ID", "Source", "Description", "Distance(mi)", "End_Lat", "End_Lng", "Wind_Chill(F)", "Precipitation(in)"]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    df.ffill(inplace=True).bfill(inplace=True)
    df.drop_duplicates(inplace=True)
    
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], format='mixed', errors='coerce')
    df["End_Time"] = pd.to_datetime(df["End_Time"], format='mixed', errors='coerce')
    df.dropna(subset=['Start_Time', 'End_Time'], inplace=True)
    
    df["hour"] = df["Start_Time"].dt.hour
    df["day"] = df["Start_Time"].dt.day
    df["month"] = df["Start_Time"].dt.month
    df["year"] = df["Start_Time"].dt.year
    df["duration"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds()
    df.drop(columns=["Start_Time", "End_Time", "Weather_Timestamp"], inplace=True, errors='ignore')
    
    print("🔑 Loading Preprocessors (Scaler/Encoders)...")
    scaler = joblib.load("models/scaler.pkl")
    le_dict = joblib.load("models/label_encoders.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        if col in le_dict:
            le = le_dict[col]
            df[col] = df[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
            
    bool_cols = df.select_dtypes(include='bool').columns
    for col in bool_cols:
        df[col] = df[col].astype(int)
        
    num_cols = df.select_dtypes(include=['int64', 'float64', 'int32']).columns.drop("Severity", errors='ignore')
    df[num_cols] = scaler.transform(df[num_cols])
    
    X = df[feature_columns]
    y = df["Severity"]
    
    print("🌲 Booting Model for Partial Fit (Online Learning)")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Did download_from_drive.py run?")
        sys.exit(1)
        
    model = joblib.load(MODEL_PATH)
    
    # Assess performance BEFORE training blindly
    print("📊 Evaluating base performance on current batch...")
    y_pred_initial = model.predict(X)
    baseline_acc = accuracy_score(y, y_pred_initial)
    print(f"Baseline Accuracy on fresh data: {baseline_acc:.4f}")
    
    # Train
    model.warm_start = True
    model.n_estimators += 20
    
    print(f"🌲 Training {model.n_estimators} total estimators...")
    model.fit(X, y)
    
    print("📊 Evaluating NEW performance...")
    y_pred_new = model.predict(X)
    new_acc = accuracy_score(y, y_pred_new)
    print(f"New Accuracy: {new_acc:.4f}")
    
    # Validation logic
    if new_acc > progress["best_accuracy"] or new_acc >= baseline_acc:
        print("💡 Model improved or remained stable! Saving updates...")
        joblib.dump(model, MODEL_PATH)
        
        if new_acc > progress["best_accuracy"]:
            progress["best_accuracy"] = new_acc
    else:
        print("⚠️ Model degraded. Skipping save to preserve existing tree matrix.")
        # We do NOT dump the model, so upload_to_drive uploads the unchanged original file!

if __name__ == "__main__":
    progress = load_progress()
    data_chunk = fetch_data_chunk(progress)
    process_and_train(data_chunk, progress)
    save_progress(progress)
    print("🎉 Training step complete!")
