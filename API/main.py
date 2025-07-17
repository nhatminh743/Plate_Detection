from fastapi import FastAPI, UploadFile, File
from typing import List
import os
from datetime import datetime
import uuid
from pathlib import Path
from datetime import datetime

#########################       DECLARE PATHS       #################################

#BASE_DIR =


#########################    END OF DECLARE PATHS   #################################
app = FastAPI()

def create_unique_folder(filename, base_dir="saved_uploads"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    folder_path = os.path.join(base_dir, f"{filename}_{timestamp}_{unique_id}")
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    folder = create_unique_folder(file.filename)
    file_path = os.path.join(folder, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"message": f"Upload file successfully, located at: {folder}"}




