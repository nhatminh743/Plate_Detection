from Extract_Letter_From_Plate.Functions.YOLO_read_func.show_result import PlotImageS
from Test_models.report_function import compare_txt_files

model_dir = '/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_Detect_Number_From_Plate/runs/content/runs/detect/train2/weights/best.pt'
image_dir = r'/home/minhpn/Desktop/Green_Parking/Test_models/YOLO_YOLO/Extracted_Plate_Data'
output_dir = r'/home/minhpn/Desktop/Green_Parking/Test_models/YOLO_YOLO/Visualization'

test_dir = '/home/minhpn/Desktop/Green_Parking/Test_models/Validation.txt'
validate_dir = '/home/minhpn/Desktop/Green_Parking/Test_models/YOLO_YOLO/Final_Result/ocr_results.txt'

list_of_mismatch = compare_txt_files(test_dir, validate_dir, long_report=True, saved=True)
plot_image_func = PlotImageS(model_dir, image_dir, output_dir, list_of_selection=list_of_mismatch, plot_selection=True)


plot_image_func.plot_all()