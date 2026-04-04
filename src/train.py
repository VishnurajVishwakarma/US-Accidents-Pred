import pandas as pd
import numpy as np
import joblib
import os
import json
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# 📥 DOWNLOAD DATASET (KAGGLE CLI)
# =========================================================
def download_dataset():
    dataset_path = "dataset/US_Accidents_March23.csv"

    if os.path.exists(dataset_path):
        print("✅ Dataset already exists")
        return dataset_path

    print("📥 Dataset not found. Downloading from Kaggle...")

    os.makedirs("dataset", exist_ok=True)

    exit_code = os.system(
        "kaggle datasets download -d sobhanmoosavi/us-accidents -p dataset --unzip"
    )

    if exit_code != 0:
        raise RuntimeError("❌ Kaggle download failed. Check API setup.")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError("❌ Dataset not found after download")

    print("✅ Dataset downloaded successfully")
    return dataset_path


# =========================================================
# 📥 LOAD DATASET (ENCODING SAFE)
# =========================================================
def load_dataset(path):
    print("📥 Loading dataset safely with chunks...")

    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
    chunk_size = 100000  # adjust if needed

    for enc in encodings:
        try:
            print(f"🔄 Trying encoding: {enc}")

            chunks = []

            for i, chunk in enumerate(pd.read_csv(
                path,
                encoding=enc,
                chunksize=chunk_size,
                on_bad_lines='skip',
                engine='python'  # more tolerant
            )):
                print(f"Loaded chunk {i+1}")
                chunks.append(chunk)

            df = pd.concat(chunks, ignore_index=True)

            print(f"Success with encoding: {enc}")
            print(f"Final Shape: {df.shape}")

            return df

        except Exception as e:
            print(f"Failed with {enc}: {e}")

    raise RuntimeError("All encoding attempts failed")


# =========================================================
# 🧹 DATA CLEANING
# =========================================================
def clean_data(df):
    print("🧹 Cleaning data...")

    drop_cols = ["ID", "Source", "Description", "Distance(mi)", "End_Lat", "End_Lng"]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    df.drop(columns=["Wind_Chill(F)", "Precipitation(in)"], inplace=True, errors='ignore')

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    df.drop_duplicates(inplace=True)

    return df


# =========================================================
# ⚙️ FEATURE ENGINEERING
# =========================================================
def feature_engineering(df):
    print("⚙️ Feature engineering...")

    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors='coerce')
    df["End_Time"] = pd.to_datetime(df["End_Time"], errors='coerce')

    df.dropna(subset=['Start_Time', 'End_Time'], inplace=True)

    df["hour"] = df["Start_Time"].dt.hour
    df["day"] = df["Start_Time"].dt.day
    df["month"] = df["Start_Time"].dt.month
    df["year"] = df["Start_Time"].dt.year
    df["duration"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds()

    df.drop(columns=["Start_Time", "End_Time", "Weather_Timestamp"], inplace=True, errors='ignore')

    return df


# =========================================================
# 🔤 ENCODING
# =========================================================
def encode_data(df):
    print("🔤 Encoding...")

    le_dict = {}

    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)

    return df, le_dict


# =========================================================
# 📊 SCALING
# =========================================================
def scale_data(df):
    print("📊 Scaling...")

    scaler = StandardScaler()

    num_cols = df.select_dtypes(include=['int64', 'float64', 'int32']).columns
    num_cols = num_cols.drop("Severity", errors='ignore')

    feature_columns = list(df.drop("Severity", axis=1).columns)

    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, scaler, feature_columns


# =========================================================
# 🌲 TRAIN MODEL
# =========================================================
def train_model(df):
    print("🌲 Training model...")

    X = df.drop("Severity", axis=1)
    y = df["Severity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("📈 Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model


# =========================================================
# 💾 SAVE ARTIFACTS
# =========================================================
def save_artifacts(model, scaler, le_dict, feature_columns):
    print("💾 Saving artifacts...")

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/accident_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(le_dict, "models/label_encoders.pkl")
    joblib.dump(feature_columns, "models/feature_columns.pkl")


# =========================================================
# 🌍 HEATMAP SAMPLE
# =========================================================
def generate_heatmap(df):
    print("🌍 Generating heatmap...")

    os.makedirs("data", exist_ok=True)

    sample = df[['Start_Lat', 'Start_Lng', 'Severity']] \
        .dropna() \
        .sample(min(3000, len(df))) \
        .to_dict(orient='records')

    with open('data/heatmap_sample.json', 'w') as f:
        json.dump(sample, f)


# =========================================================
# 🚀 MAIN
# =========================================================
def main():

    # Step 1: Download dataset if missing
    dataset_path = download_dataset()

    # Step 2: Load data
    df = load_dataset(dataset_path)

    # Step 3: Heatmap
    generate_heatmap(df)

    # Step 4: Clean
    df = clean_data(df)

    # Step 5: Feature engineering
    df = feature_engineering(df)

    # Step 6: Encode
    df, le_dict = encode_data(df)

    # Step 7: Scale
    df, scaler, feature_columns = scale_data(df)

    # Step 8: Train
    model = train_model(df)

    # Step 9: Save
    save_artifacts(model, scaler, le_dict, feature_columns)

    print("✅ PIPELINE COMPLETE")


# =========================================================
# ▶️ ENTRY
# =========================================================
if __name__ == "__main__":
    main()