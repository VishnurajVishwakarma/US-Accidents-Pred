from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import os

SCOPES = ['https://www.googleapis.com/auth/drive']
FOLDER_ID = '1bM_YxY-A1sGDzlzQdLfHTNjEgEObCddu'

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

file_id = get_file_id("accident_model.pkl")

if file_id:
    request = service.files().get_media(fileId=file_id)
    os.makedirs("models", exist_ok=True)
    fh = io.FileIO("models/accident_model.pkl", 'wb')

    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            print(f"Download {int(status.progress() * 100)}%.")

    print("Model downloaded")
else:
    print("No existing model found, starting fresh")
