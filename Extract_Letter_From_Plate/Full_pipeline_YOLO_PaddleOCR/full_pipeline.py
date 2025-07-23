from Extract_Letter_From_Plate.Functions.YOLO_plate_func import extracted_plate_YOLO
from Test_models.sort_alphabetically_txt import sort_txt_by_title
from Extract_Letter_From_Plate.Functions.utils import clear_directory
from Extract_Letter_From_Plate.Functions.PaddleOCR.paddleOCR import PaddleOCRLineExtractor

RAW_DATA_DIR = r'/home/minhpn/Desktop/Green_Parking/Dummy_Data_For_Small_Test/Raw_Data'
EXTRACTED_PLATE_DIR = r'/home/minhpn/Desktop/Green_Parking/Dummy_Data_For_Small_Test/Extracted_Plate_Data'
EXTRACTED_LINE_DIR = r'/home/minhpn/Desktop/Green_Parking/Dummy_Data_For_Small_Test/Extracted_Line_DIr'
FINAL_RESULT_DIR = r'/home/minhpn/Desktop/Green_Parking/Test_models/YOLO_YOLO/Final_Result'
YOLO_plate_dir = r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_training/runs/detect/train2/weights/best.pt'

def main_pipeline():

    # Step 0: Clear directory

    clear_directory(EXTRACTED_PLATE_DIR)
    clear_directory(FINAL_RESULT_DIR)

    # Step 1: Plate Extraction
    extractor = extracted_plate_YOLO.PlateExtractor(
        data_dir=RAW_DATA_DIR,
        save_dir=EXTRACTED_PLATE_DIR,
        best_model_file= YOLO_plate_dir,
    )
    extractor.process_images()

    read = PaddleOCRLineExtractor(
        data_dir=EXTRACTED_PLATE_DIR,
        save_dir=EXTRACTED_PLATE_DIR,
    )

