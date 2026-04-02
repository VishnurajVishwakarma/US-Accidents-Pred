import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')
import kagglehub
from kagglehub import KaggleDatasetAdapter

MODEL_PATH = "models/accident_model.pkl"

def fetch_data_chunk():
    print("🔄 Fetching Kaggle Dataset...")
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "sobhanmoosavi/us-accidents",
        "US_Accidents_March23.csv"
    )
    chunk = df.sample(frac=0.4, random_state=None)
    print(f"✅ Loaded 40% random chunk. Shape: {chunk.shape}")
    return chunk

def process_and_train(df):
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
    model.warm_start = True
    model.n_estimators += 20
    
    print(f"🌲 Training {model.n_estimators} total estimators...")
    model.fit(X, y)
    
    print("💾 Saving updated model...")
    joblib.dump(model, MODEL_PATH)

if __name__ == "__main__":
    data_chunk = fetch_data_chunk()
    process_and_train(data_chunk)
    print("🎉 Training step complete!")
