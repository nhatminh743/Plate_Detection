from Extract_Letter_From_Plate.Functions.YOLO_plate_func import extracted_plate_YOLO
from Test_models.sort_alphabetically_txt import sort_txt_by_title
from Extract_Letter_From_Plate.Functions.utils import clear_directory
from Extract_Letter_From_Plate.Functions.PaddleOCR.paddleOCR import PaddleOCRLineExtractor

RAW_DATA_DIR = r'/home/minhpn/Desktop/Green_Parking/one_image/saved'
EXTRACTED_PLATE_DIR = r'/home/minhpn/Desktop/Green_Parking/one_image/Extracted_Plate_Data'
EXTRACTED_LINE_DIR = r'/home/minhpn/Desktop/Green_Parking/one_image/Extracted_Line'
FINAL_RESULT_DIR = r'/home/minhpn/Desktop/Green_Parking/one_image/Final_Result'
YOLO_plate_dir = r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_training/runs/detect/train2/weights/best.pt'
TEXT_RECOGNITION_DIR = r'/home/minhpn/Desktop/Green_Parking/Model_training/PaddleOCR_finetune/content/PaddleOCR/output/inference/PP-OCRv5_server_rec'

def main_pipeline():

    # Step 0: Clear directory

    clear_directory(EXTRACTED_PLATE_DIR)
    clear_directory(FINAL_RESULT_DIR)
    clear_directory(EXTRACTED_LINE_DIR)

    # Step 1: Plate Extraction
    extractor = extracted_plate_YOLO.PlateExtractor(
        data_dir=RAW_DATA_DIR,
        save_dir=EXTRACTED_PLATE_DIR,
        best_model_file= YOLO_plate_dir,
    )
    extractor.process_images()

    read = PaddleOCRLineExtractor(
        data_dir=EXTRACTED_PLATE_DIR,
        save_dir=FINAL_RESULT_DIR,
        temporary_dir=EXTRACTED_LINE_DIR,
        text_recognition_dir=TEXT_RECOGNITION_DIR
    )
    read.run()

if __name__ == "__main__":
    main_pipeline()
