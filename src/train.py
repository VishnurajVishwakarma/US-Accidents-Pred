import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

import sys

def load_dataset(fallback_path):
    try:
        print("🔄 Trying KaggleHub...")
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "sobhanmoosavi/us-accidents",
            "US_Accidents_March23.csv"
        )

        print("✅ Loaded using KaggleHub")
        return df

    except Exception as e:
        print("⚠️ KaggleHub failed:", e)
        print("🔄 Switching to CSV fallback...")

        chunk_size = 100000
        chunks = []

        for chunk in pd.read_csv(
            fallback_path,
            chunksize=chunk_size,
            low_memory=False
        ):
            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)

        print("✅ Loaded using CSV fallback")
        return df

def main():
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "data/US_Accidents.csv"
        if not os.path.exists(data_path) and os.path.exists("../data/US_Accidents.csv"):
            data_path = "../data/US_Accidents.csv"
            
    # 🔥 Use it
    df = load_dataset(data_path)
        
    print(f"Data shape after loading: {df.shape}")

    # Save a heatmap sample before cleaning corrupts original lats/longs
    print("Generating heatmap sample...")
    os.makedirs("data", exist_ok=True)
    heatmap_sample = df[['Start_Lat', 'Start_Lng', 'Severity']].dropna().sample(min(3000, len(df))).to_dict(orient='records')
    with open('data/heatmap_sample.json', 'w') as f:
        json.dump(heatmap_sample, f)

    print("Cleaning data...")
    # Drop irrelevant columns
    drop_cols = ["ID", "Source", "Description", "Distance(mi)", "End_Lat", "End_Lng"]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    # Handle missing values
    df.drop(columns=["Wind_Chill(F)", "Precipitation(in)"], inplace=True, errors='ignore')
    
    # Fill remaining nulls
    df.ffill(inplace=True)
    df.bfill(inplace=True) 
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Fix Data Types
    print("Feature engineering...")
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], format='mixed', errors='coerce')
    df["End_Time"] = pd.to_datetime(df["End_Time"], format='mixed', errors='coerce')
    
    df.dropna(subset=['Start_Time', 'End_Time'], inplace=True)
    
    # Extract time features
    df["hour"] = df["Start_Time"].dt.hour
    df["day"] = df["Start_Time"].dt.day
    df["month"] = df["Start_Time"].dt.month
    df["year"] = df["Start_Time"].dt.year
    df["duration"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds()
    
    # Drop original time cols
    df.drop(columns=["Start_Time", "End_Time", "Weather_Timestamp"], inplace=True, errors='ignore')
    
    # Encode categorical variables
    print("Encoding categorical variables...")
    le_dict = {}
    cat_cols = df.select_dtypes(include='object').columns
    
    for col in cat_cols:
        le = LabelEncoder()
        # Convert to string to avoid mixed type errors
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
        
    # Also encode boolean cols if any
    bool_cols = df.select_dtypes(include='bool').columns
    for col in bool_cols:
        df[col] = df[col].astype(int)
        
    # Scale numerical
    print("Scaling features...")
    scaler = StandardScaler()
    # exclude Severity from scaling
    num_cols = df.select_dtypes(include=['int64', 'float64', 'int32']).columns.drop("Severity", errors='ignore')
    
    # Ensure column order is saved for inference
    feature_columns = list(df.drop("Severity", axis=1).columns)
    
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Split
    print("Splitting data...")
    X = df.drop("Severity", axis=1)
    y = df["Severity"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Save model
    os.makedirs("models", exist_ok=True)
    print("Saving model and preprocessors...")
    joblib.dump(model, "models/accident_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(le_dict, "models/label_encoders.pkl")
    joblib.dump(feature_columns, "models/feature_columns.pkl")
    
    print("Done!")

if __name__ == "__main__":
    main()
