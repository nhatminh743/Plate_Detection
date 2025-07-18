#########################       DECLARE PATHS       #################################
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
sys.path.append(str(PARENT_DIR))
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "saved_uploads"
PLATE_DIR = STATIC_DIR / "extracted_plates"
RESULT_DIR = STATIC_DIR / "final_results"
YOLO_plate_model = r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_training/runs/detect/train3/weights/best.pt'
YOLO_read_model = r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_Detect_Number_From_Plate/runs/content/runs/detect/train2/weights/best.pt'
#########################    END OF DECLARE PATHS   #################################
#######################      IMPORT LIBRARIES     ################################

from fastapi import FastAPI, UploadFile, File
from typing import List
from datetime import datetime
import os
from fastapi.staticfiles import StaticFiles
from Extract_Letter_From_Plate.Functions.utils import clear_directory
from Extract_Letter_From_Plate.Functions.YOLO_plate_func import extracted_plate_YOLO
from Extract_Letter_From_Plate.Functions.YOLO_read_func import letter_YOLO
from Test_models.sort_alphabetically_txt import sort_txt_by_title


########################        DECLARE APP        ##################################
app = FastAPI()

#########################            END            #################################
#########################     CREATE DIR IF NOT     #################################

for d in [UPLOAD_DIR, PLATE_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

#########################            END            #################################

##########################     ALLOW USER TO ACCESS     #############################
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


###########################       FUNCTIONS         ##################################

def create_unique_folder(filename, base_dir=UPLOAD_DIR):
    now = datetime.now()
    time_now = now.strftime('%Y-%m-%d_%H-%M-%S')
    folder_path = os.path.join(base_dir, f"{filename}_{time_now}")
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    pure_filename = extract_pure_name(file.filename)
    folder = create_unique_folder(pure_filename)
    file_path = os.path.join(folder, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    print(f"Upload file successfully, located at: {folder}")

    session_path = folder

    clear_directory(str(PLATE_DIR))
    clear_directory(str(RESULT_DIR))

    # Step 1: Detect plates
    extractor = extracted_plate_YOLO.PlateExtractor(
        data_dir=str(session_path),
        save_dir=str(PLATE_DIR),
        best_model_file=str(YOLO_plate_model),
    )
    extractor.process_images()

    # Step 2: Read characters
    reader = letter_YOLO.LetterExtractor(
        data_dir=str(PLATE_DIR),
        save_dir=str(RESULT_DIR),
        best_model_file=str(YOLO_read_model),
        debug_mode=False,
    )
    reader.process_images()

    # Step 3: Sort results
    result_txt = RESULT_DIR / "ocr_results.txt"
    if not(result_txt.exists()):
        os.makedirs(str(result_txt), exist_ok=True)

    sort_txt_by_title(str(result_txt))

    # Step 4: Load results into JSON
    results = {}
    if result_txt.exists():
        with open(result_txt, "r") as f:
            for line in f:
                if ':' in line:
                    img_name, ocr_text = line.strip().split(':', 1)
                    plate_path = PLATE_DIR / f"{img_name.strip()[:12]}.jpg"
                    results[img_name.strip()[:12]] = {
                        "text": ocr_text.strip()
                    }

    return {"results": results}

################################### END OF MAIN FUNCTION #####################################################
##########################################      UTILS     ####################################################

def extract_pure_name(filename):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            return filename[:-4]
        elif filename.endswith('.jpeg'):
            return filename[:-5]
    elif filename.endswith('.zip'):
        pass
    else:
        return {'error': 'Only accept file type of .jpg, .png, .jpeg or .zip'}

