# import the required libraries
from __future__ import print_function
from datetime import time
from http.cookiejar import LoadError
import os.path
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError


class DriveAPI:
    global SCOPES
    # Define the scopes
    SCOPES = ["https://www.googleapis.com/auth/drive.file"]
    def __init__(self, credentials_path, token_path):
        self.credentials_path = credentials_path
        self.token_path = token_path

        self.creds = None
        if os.path.exists(self.token_path):
            self.creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES)
                self.creds = flow.run_local_server(port=0)
            with open(self.token_path, "w") as token:
                token.write(self.creds.to_json())

        # Connect to the API service
        self.service = build('drive', 'v3', credentials=self.creds)

    #  Return folder_id if exists, else return None
    def get_folder_id_if_exists(self, folder_name, parent_folder_id=None):
        query = (
            f"name = '{folder_name}' "
            f"and mimeType = 'application/vnd.google-apps.folder' "
            f"and trashed = false"
        )
        if parent_folder_id:
            query += f" and '{parent_folder_id}' in parents"
        results = self.service.files().list(
            q=query,
            fields="files(id, name)",
            pageSize=1
        ).execute()
        folders = results.get("files", [])
        if folders:
            return folders[0]["id"]
        return None
    
    # Return fileId if file exists in parent folder, else None
    def find_file_in_folder(self, filename, parent_folder_id):
        query = (
            f"name = '{filename}' "
            f"and '{parent_folder_id}' in parents "
            f"and trashed = false"
        )
        results = self.service.files().list(
            q=query,
            fields="files(id, name)",
            pageSize=1
        ).execute()      
        files = results.get("files", [])
        return files[0]["id"] if files else None

    # Create the folder name 
    def create_folder(self, folder_name, parent_folder_id=None):
        try:
            metadata = {   
            "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
            }
            if parent_folder_id:
                metadata["parents"] = [parent_folder_id]

            folder = self.service.files().create(
                body=metadata,
                fields="id"
            ).execute()
            return folder["id"]

        except HttpError as error:
            print(f"An error occurred: {error}")
            return None
        

    # If folder exists -> return id. If not -> create and return id
    def ensure_folder(self, folder_name, parent_folder_id=None):
        folder_id = self.get_folder_id_if_exists(
            folder_name,
            parent_folder_id
        )
        if folder_id:
            print(f"Found folder '{folder_name}' (id={folder_id})")
            return folder_id
        
        print(f"Creating folder '{folder_name}' ...")
        return self.create_folder(
            folder_name,
            parent_folder_id
        )

    # Upload file (image/txt/csv)
    def FileUpload(self, filepath, parent_folder_id, mime_type='application/octet-stream'):
        file_metadata = {
            'name': os.path.basename(filepath),
            'parents': [parent_folder_id]
        } # create name using basename
        try:
            media = MediaFileUpload(filepath, mimetype=mime_type)
            # MediaFileUpload(filepath, mimetype=mime_type, resumable=True) #if file > 5MB
            file = self.service.files().create(
                body=file_metadata, media_body=media, fields='id').execute()
            print(f'File uploaded with ID: {file.get("id")}')
            
        except TimeoutError:
            print("Timeout Error!")
            time.sleep(2)
        except Exception as e:
            print(f"Upload failed: {e}")


        
    # Update file (only csv) if file existed using fileId
    def overwrite_file(self, file_id, filepath, mime_type="text/csv"):
        media = MediaFileUpload(
            filepath,
            mimetype=mime_type,
            resumable=True
        )
        try:
            updated = self.service.files().update(
                fileId=file_id,
                media_body=media,
                fields="id, modifiedTime"
            ).execute()
            print(f"File overwritten: {updated['id']}")
            return updated

        except HttpError as e:
            raise RuntimeError(f"Overwrite failed: {e}")
        
    
    # If file exists -> return id, overwrite this file. If not -> create and upload it
    def upload_or_overwrite(self, filepath, parent_folder_id, mime_type="text/csv"):
        filename = os.path.basename(filepath)
        file_id = self.find_file_in_folder(filename, parent_folder_id)
        if file_id:
            print("File exists → overwrite")
            return self.overwrite_file(file_id, filepath, mime_type)
        else:
            print("File not found → create new")
            return self.FileUpload(filepath, parent_folder_id)

if __name__ == "__main__":
    obj = DriveAPI()
    cloud_name = obj.ensure_folder(folder_name="Inspectionsystemcloud", parent_folder_id=None)
    csv_name = obj.ensure_folder(folder_name="csv", parent_folder_id=cloud_name)
    image_name = obj.ensure_folder(folder_name="image", parent_folder_id=cloud_name)
   
    # Upload the new log or overwrite the existed log
    """obj.upload_or_overwrite(
        filepath="summary/Images/2026-03-26.csv",
        parent_folder_id=csv_name,
        mime_type="text/csv"
    )"""
    # Upload image
    obj.FileUpload(filepath="11-39-16_175_0.jpg", parent_folder_id=image_name)
    
