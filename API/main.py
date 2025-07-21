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
PLOT_DIR = STATIC_DIR / "plotted_images"

'''
BIG NOTE: LINUX dir
'''
YOLO_plate_model = r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_training/runs/detect/train2/weights/best.pt'
YOLO_read_model = r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_Detect_Number_From_Plate/runs/content/runs/detect/train2/weights/best.pt'

'''
BIGNOTE: WINDOW DIR

USER: REPLACE YOUR CURRENT DIRECTORY IN ORDER FOR THIS TO RUN.
'''

# YOLO_plate_model = r'C:\Users\ACER\Documents\nhatminh743\Plate_Detection\Model_training\YOLOv11_training\runs\detect\train3\weights\best.pt'
# YOLO_read_model = r'C:\Users\ACER\Documents\nhatminh743\Plate_Detection\Model_training\YOLOv11_Detect_Number_From_Plate\runs\content\runs\detect\train2\weights\best.pt'


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
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import base64

########################        DECLARE APP        ##################################
app = FastAPI()

##########################################      UTILS     ####################################################

def extract_pure_name(filename):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            return filename[:-4]
        elif filename.endswith('.jpeg'):
            return filename[:-5]
    # elif filename.endswith('.zip'):
    else:
        return {'error': 'Only accept file type of .jpg, .png, .jpeg'}

def create_unique_folder(filename, base_dir=UPLOAD_DIR):
    now = datetime.now()
    time_now = now.strftime('%Y-%m-%d_%H-%M-%S')
    folder_path = os.path.join(base_dir, f"{filename}_{time_now}")
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

#########################     CREATE IMPORTANT DIR IF NOT     #################################

for d in [UPLOAD_DIR, PLATE_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

##########################     ALLOW USER TO ACCESS     #############################

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

###########################       FUNCTIONS         ##################################

@app.post('/upload-files-multiple')
async def upload_files_multiple(files: List[UploadFile] = File(...)):
    pure_filename = extract_pure_name(files[0].filename)
    folder = create_unique_folder(pure_filename)

    CURR_PLATE_DIR = create_unique_folder(pure_filename, base_dir=PLATE_DIR)
    CURR_RESULT_DIR = create_unique_folder(pure_filename, base_dir=RESULT_DIR)

    for file in files:
        file_path = os.path.join(folder, file.filename)
        with open(file_path, 'wb') as f:
            f.write(await file.read())
            print(f'Saved: {file.filename}')

    req = ProcessRequest(
        session_path=folder,
        CURR_PLATE_DIR=CURR_PLATE_DIR,
        CURR_RESULT_DIR=CURR_RESULT_DIR,
    )

    result = process_uploaded_folder(req)

    return result

@app.post('/upload-files-single')
async def upload_files_single(file: UploadFile = File(...)):
    pure_filename = extract_pure_name(file.filename)
    folder = create_unique_folder(pure_filename)

    CURR_PLATE_DIR = create_unique_folder(pure_filename, base_dir=PLATE_DIR)
    CURR_RESULT_DIR = create_unique_folder(pure_filename, base_dir=RESULT_DIR)

    file_path = os.path.join(folder, file.filename)

    contents = await file.read()

    with open(file_path, 'wb') as f:
        f.write(contents)
    print(f'Saved: {file.filename}')

    req = ProcessRequest(
        session_path=folder,
        CURR_PLATE_DIR=CURR_PLATE_DIR,
        CURR_RESULT_DIR=CURR_RESULT_DIR,
    )

    image = Image.open(io.BytesIO(contents))

    result, dimension, org_dim, model_dim = process_uploaded_folder(req)
    x1, y1, x2, y2 = dimension
    fig, ax = plt.subplots()
    ax.imshow(image)

    # rect = patches.Rectangle(
    #     (x, y), w, h, linewidth=1, edgecolor='red', facecolor='none'
    # )
    # ax.add_patch(rect)

    polygon_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    polygon = patches.Polygon(polygon_points, closed=True, edgecolor='blue', linewidth=2, facecolor='none')
    ax.add_patch(polygon)

    label = list(list(result.values())[0].values())[0]
    ax.text(
        x2 + 3, y1 - 5,
        label,
        fontsize=10,
        color='white',
        bbox=dict(facecolor='blue', edgecolor='none', boxstyle='round,pad=0.2')
    )

    plt.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    img_bytes = buf.getvalue()
    base64_image = base64.b64encode(img_bytes).decode('utf-8')
    img_data_url = f"data:image/png;base64,{base64_image}"

    return JSONResponse(content={
        "text": result,
        "dimension_of_plate": dimension,
        'org_dim': org_dim,
        'model_dim': model_dim,
        "image": img_data_url
    })

class ProcessRequest(BaseModel):
    session_path: str  # The path returned by `/upload-file`
    CURR_PLATE_DIR: str
    CURR_RESULT_DIR: str



@app.post("/process-folder")
def process_uploaded_folder(req: ProcessRequest):
    session_path = Path(req.session_path)
    CURR_PLATE_DIR = Path(req.CURR_PLATE_DIR)
    CURR_RESULT_DIR = Path(req.CURR_RESULT_DIR)

    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Session path does not exist.")

    if not os.path.exists(CURR_PLATE_DIR):
        raise HTTPException(status_code=404, detail="Current direction for saving plate does not exist.")

    if not os.path.exists(CURR_RESULT_DIR):
        raise HTTPException(status_code=404, detail="Current direction for saving result does not exist.")

    # Clear previous results
    clear_directory(str(CURR_PLATE_DIR))
    clear_directory(str(CURR_RESULT_DIR))

    # Step 1: Detect plates
    extractor = extracted_plate_YOLO.PlateExtractor(
        data_dir=str(session_path),
        save_dir=str(CURR_PLATE_DIR),
        best_model_file=str(YOLO_plate_model),
    )
    dimension, ori_dimension, model_dimension = extractor.process_images()

    # Step 2: Read characters
    reader = letter_YOLO.LetterExtractor(
        data_dir=str(CURR_PLATE_DIR),
        save_dir=str(CURR_RESULT_DIR),
        best_model_file=str(YOLO_read_model),
        debug_mode=False,
    )
    reader.process_images()

    # Step 3: Sort OCR results
    result_txt = CURR_RESULT_DIR / "ocr_results.txt"
    result_txt.touch(exist_ok=True)
    sort_txt_by_title(str(result_txt))

    # Step 4: Format result into JSON
    results = {}

    print("Opening result file:", result_txt)

    with open(result_txt, "r", encoding="utf-8") as f:
        for line in f:
            print(f"[Raw Line] {repr(line)}")  # <- show full line content including \n etc

            if ':' not in line:
                print("  [Skipped] No colon found")
                continue

            try:
                img_name, ocr_text = line.strip().split(':', 1)
                short_name = Path(img_name.strip()).stem
                results[short_name] = {
                    "text": ocr_text.strip()
                }
                print(f"  [Parsed] {short_name}: {ocr_text.strip()}")
            except Exception as e:
                print("  [Error parsing line]", e)

    print("Final results:", results)
    print("Final dimension:", dimension[0])

    return results, dimension[0], ori_dimension, model_dimension



################################### END OF MAIN FUNCTION #####################################################


#######################################      EXCESS CODE      ########################################

# @app.post("/upload-file")
# async def upload_file(file: UploadFile = File(...)):
#     pure_filename = extract_pure_name(file.filename)
#     folder = create_unique_folder(pure_filename)
#     file_path = os.path.join(folder, file.filename)
#     with open(file_path, "wb") as f:
#         f.write(await file.read())
#     print(f"Upload file successfully, located at: {folder}")
#
#     session_path = folder
#
#     clear_directory(str(PLATE_DIR))
#     clear_directory(str(RESULT_DIR))
#
#     # Step 1: Detect plates
#     extractor = extracted_plate_YOLO.PlateExtractor(
#         data_dir=str(session_path),
#         save_dir=str(PLATE_DIR),
#         best_model_file=str(YOLO_plate_model),
#     )
#     extractor.process_images()
#
#     # Step 2: Read characters
#     reader = letter_YOLO.LetterExtractor(
#         data_dir=str(PLATE_DIR),
#         save_dir=str(RESULT_DIR),
#         best_model_file=str(YOLO_read_model),
#         debug_mode=False,
#     )
#     reader.process_images()
#
#     # Step 3: Sort results
#     result_txt = RESULT_DIR / "ocr_results.txt"
#
#     if not result_txt.exists():
#         result_txt.touch()
#
#     sort_txt_by_title(str(result_txt))
#
#     # Step 4: Load results into JSON
#     results = {}
#     if result_txt.exists():
#         with open(result_txt, "r") as f:
#             for line in f:
#                 if ':' in line:
#                     img_name, ocr_text = line.strip().split(':', 1)
#                     plate_path = PLATE_DIR / f"{img_name.strip()[:12]}.jpg"
#                     results[img_name.strip()[:12]] = {
#                         "text": ocr_text.strip()
#                     }
#
#     return {"results": results}