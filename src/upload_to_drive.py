from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
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
        print("⚠️ No GCP credentials found! Skipping upload.")
        sys.exit(0)
except Exception as e:
    print(f"❌ Failed to parse GCP credentials: {e}")
    sys.exit(0)

service = build('drive', 'v3', credentials=creds)

def get_file_id(name):
    results = service.files().list(
        q=f"name='{name}' and '{FOLDER_ID}' in parents and trashed=false",
        fields="files(id, name)"
    ).execute()
    files = results.get('files', [])
    return files[0]['id'] if files else None

for file_name in FILES_TO_SYNC:
    local_path = f"models/{file_name}"
    if not os.path.exists(local_path):
        print(f"{local_path} not found locally. Skipping upload.")
        continue
        
    file_id = get_file_id(file_name)
    media = MediaFileUpload(local_path, resumable=True)

    if file_id:
        print(f"Updating existing {file_name} in Drive...")
        service.files().update(fileId=file_id, media_body=media).execute()
    else:
        print(f"Uploading new {file_name} to Drive...")
        service.files().create(
            body={'name': file_name, 'parents': [FOLDER_ID]},
            media_body=media
        ).execute()
