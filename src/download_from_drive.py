from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import os
import sys

SCOPES = ['https://www.googleapis.com/auth/drive']
FOLDER_ID = '1bM_YxY-A1sGDzlzQdLfHTNjEgEObCddu'
FILES_TO_SYNC = ["accident_model.pkl", "progress.json"]

try:
    if "GCP_SA_KEY" in os.environ:
        import json
        creds_json = json.loads(os.environ["GCP_SA_KEY"])
        creds = service_account.Credentials.from_service_account_info(creds_json, scopes=SCOPES)
    elif os.path.exists('gcp-key.json'):
        creds = service_account.Credentials.from_service_account_file('gcp-key.json', scopes=SCOPES)
    else:
        print("⚠️ No GCP credentials found! Skipping download.")
        sys.exit(0)
except Exception as e:
    print(f"❌ Failed to parse GCP credentials: {e}")
    sys.exit(0)  # Exit safely so Gunicorn still boots

service = build('drive', 'v3', credentials=creds)

def get_file_id(name):
    results = service.files().list(
        q=f"name='{name}' and '{FOLDER_ID}' in parents and trashed=false",
        fields="files(id, name)"
    ).execute()
    files = results.get('files', [])
    return files[0]['id'] if files else None

os.makedirs("models", exist_ok=True)

for file_name in FILES_TO_SYNC:
    if os.path.exists(f"models/{file_name}"):
        print(f"{file_name} already exists locally. Skipping download.")
        continue

    file_id = get_file_id(file_name)
    if file_id:
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(f"models/{file_name}", 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        print(f"{file_name} downloaded successfully.")
    else:
        print(f"No existing {file_name} found in Drive, starting fresh.")
