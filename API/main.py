from fastapi import FastAPI, UploadFile, File
from typing import List
import os
from datetime import datetime
import uuid

app = FastAPI()

def create_unique_folder(base_dir="uploads"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    folder_path = os.path.join(base_dir, f"{timestamp}_{unique_id}")
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    folder = create_unique_folder()
    file_path = os.path.join(folder, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"message": "Single file uploaded", "folder": folder}

@app.post("/upload-folder")
async def upload_folder(files: List[UploadFile] = File(...)):
    folder = create_unique_folder()
    filenames = []

    for file in files:
        file_path = os.path.join(folder, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        filenames.append(file.filename)

    return {"message": f"{len(filenames)} files uploaded", "folder": folder}
