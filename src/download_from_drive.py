from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
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

os.makedirs("models", exist_ok=True)

for file_name in FILES_TO_SYNC:
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
