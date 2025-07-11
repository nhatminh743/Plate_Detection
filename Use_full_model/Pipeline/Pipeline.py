import Extract_Letter_From_Plate.Functions as F
import extract_plate

# Paths
# RAW_DATA_DIR = r'/home/minhpn/Desktop/Green_Parking/Small_Dummy_Data/Raw_Data'
# EXTRACTED_PLATE_DIR = r'/home/minhpn/Desktop/Green_Parking/Small_Dummy_Data/Extracted_Plate_Data'
# FINAL_RESULT_DIR = r'/home/minhpn/Desktop/Green_Parking/Small_Dummy_Data/Final_Result'

RAW_DATA_DIR = r'/home/minhpn/Desktop/Green_Parking/Test/Data'
EXTRACTED_PLATE_DIR = r'/home/minhpn/Desktop/Green_Parking/Test/YOLO_EasyOCR/Extracted_Plate_Data'
FINAL_RESULT_DIR = r'/home/minhpn/Desktop/Green_Parking/Test/YOLO_EasyOCR/Final_Result'

def main_pipeline():
    # Step 1: Plate Extraction
    extractor = extract_plate.PlateExtractor(
        data_dir=RAW_DATA_DIR,
        save_dir=EXTRACTED_PLATE_DIR,
        best_model_file= '/home/minhpn/Desktop/Green_Parking/Use_full_model/runs/detect/train2/weights/best.pt',
    )
    extractor.process_images()

    # Step 2: OCR Processing
    ocr_processor = F.PlateOCRProcessor(
        data_dir=EXTRACTED_PLATE_DIR,
        save_dir=FINAL_RESULT_DIR
    )
    ocr_processor.process_images()

if __name__ == '__main__':
    main_pipeline()
