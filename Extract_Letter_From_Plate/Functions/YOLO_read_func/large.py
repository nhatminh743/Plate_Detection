from letter_YOLO import LetterExtractor

best_model_dir = r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_Detect_Number_From_Plate/runs/content/runs/detect/train2/weights/best.pt'
data_dir = r'/home/minhpn/Desktop/Green_Parking/Dummy_Data_For_Small_Test/Extracted_Plate_Data'
save_dir = r'/home/minhpn/Desktop/Green_Parking/Dummy_Data_For_Small_Test/Extracted_Letter_Data'
new = LetterExtractor(
    data_dir=data_dir,
    save_dir=save_dir,
    best_model_file=best_model_dir
                      )
new.process_images()

