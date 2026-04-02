from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

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

media = MediaFileUpload("models/accident_model.pkl", resumable=True)

if file_id:
    print("Updating existing model...")
    service.files().update(fileId=file_id, media_body=media).execute()
else:
    print("Uploading new model...")
    service.files().create(
        body={'name': 'accident_model.pkl', 'parents': [FOLDER_ID]},
        media_body=media
    ).execute()
