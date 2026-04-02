import pandas as pd
import numpy as np
import joblib
import os
import sys
import tempfile
from sklearn.ensemble import RandomForestClassifier

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import io

import warnings
warnings.filterwarnings('ignore')
import kagglehub
from kagglehub import KaggleDatasetAdapter

DRIVE_FOLDER_ID = "1bM_YxY-A1sGDzlzQdLfHTNjEgEObCddu"
CREDENTIALS_FILE = "gcp-key.json"
MODEL_PATH = "models/accident_model.pkl"

def authenticate_drive():
    print("🔒 Authenticating with Google Drive...")
    if not os.path.exists(CREDENTIALS_FILE):
        print(f"Error: {CREDENTIALS_FILE} not found. Check GitHub Secrets.")
        sys.exit(1)
        
    scopes = ['https://www.googleapis.com/auth/drive']
    creds = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=scopes)
    return build('drive', 'v3', credentials=creds)

def download_model_from_drive(service):
    print("☁️ Searching for accident_model.pkl in Google Drive...")
    results = service.files().list(q=f"'{DRIVE_FOLDER_ID}' in parents and name='accident_model.pkl' and trashed=false", fields="files(id, name)").execute()
    items = results.get('files', [])
    
    if not items:
        print("Model not found in Drive. Aborting online training.")
        sys.exit(1)
        
    file_id = items[0]['id']
    print(f"📥 Downloading model from Drive (ID: {file_id})...")
    
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(MODEL_PATH, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            print(f"Download {int(status.progress() * 100)}%.")
    return file_id

def upload_model_to_drive(service, file_id):
    print(f"📤 Uploading updated model back to Google Drive...")
    media = MediaFileUpload(MODEL_PATH, mimetype='application/octet-stream')
    service.files().update(fileId=file_id, media_body=media).execute()
    print("✅ Upload complete!")

def fetch_data_chunk():
    print("🔄 Fetching Kaggle Dataset...")
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "sobhanmoosavi/us-accidents",
        "US_Accidents_March23.csv"
    )
    # Target definition - 40% random partial strategy
    chunk = df.sample(frac=0.4, random_state=None)
    print(f"✅ Loaded 40% random chunk. Shape: {chunk.shape}")
    return chunk

def process_and_train(df):
    print("⚙️ Processing chunk...")
    # Clean data (Same as train.py)
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
    
    # Handle unseen categories gracefully
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        if col in le_dict:
            le = le_dict[col]
            # Map unseen valid items to the first class to avoid crash during encode
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
    model = joblib.load(MODEL_PATH)
    
    # Warm start to allow adding new trees incrementally without destroying previous knowledge
    model.warm_start = True
    model.n_estimators += 20  # Append 20 new trees learning from this 40% dataset slice
    
    print(f"🌲 Training {model.n_estimators} total estimators...")
    model.fit(X, y)
    
    print("💾 Saving updated model...")
    joblib.dump(model, MODEL_PATH)

if __name__ == "__main__":
    service = authenticate_drive()
    file_id = download_model_from_drive(service)
    
    data_chunk = fetch_data_chunk()
    process_and_train(data_chunk)
    
    upload_model_to_drive(service, file_id)
    print("🎉 Online Learning Cycle Complete!")
