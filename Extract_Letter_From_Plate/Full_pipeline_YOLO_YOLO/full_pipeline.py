from Extract_Letter_From_Plate.Functions.YOLO_plate_func import extracted_plate_YOLO
from Extract_Letter_From_Plate.Functions.YOLO_read_func import letter_YOLO

RAW_DATA_DIR = r'/home/minhpn/Desktop/Green_Parking/Test_models/Data'
EXTRACTED_PLATE_DIR = r'/home/minhpn/Desktop/Green_Parking/Test_models/YOLO_YOLO/Extracted_Plate_Data'
FINAL_RESULT_DIR = r'/home/minhpn/Desktop/Green_Parking/Test_models/YOLO_YOLO/Final_Result'
YOLO_read_dir = r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_Detect_Number_From_Plate/runs/content/runs/detect/train2/weights/best.pt'
YOLO_plate_dir = r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_training/runs/detect/train3/weights/best.pt'

def main_pipeline():
    # Step 1: Plate Extraction
    extractor = extracted_plate_YOLO.PlateExtractor(
        data_dir=RAW_DATA_DIR,
        save_dir=EXTRACTED_PLATE_DIR,
        best_model_file= YOLO_plate_dir,
    )
    extractor.process_images()

    #Step 2: Read the plate
    read = letter_YOLO.LetterExtractor(
        data_dir = EXTRACTED_PLATE_DIR,
        save_dir = FINAL_RESULT_DIR,
        best_model_file = YOLO_read_dir,
    )
    read.process_images()

if __name__ == '__main__':
    main_pipeline()