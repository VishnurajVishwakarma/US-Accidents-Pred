from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os

SCOPES = ['https://www.googleapis.com/auth/drive']
FOLDER_ID = '1bM_YxY-A1sGDzlzQdLfHTNjEgEObCddu'
FILES_TO_SYNC = ["accident_model.pkl", "progress.json"]

creds = service_account.Credentials.from_service_account_file(
    'gcp-key.json', scopes=SCOPES)
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
