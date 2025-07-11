import sys
sys.path.append('/home/minhpn/Desktop/Green_Parking/Test_models')

from report import compare_txt_files

test_dir = '/home/minhpn/Desktop/Green_Parking/Test_models/Validation.txt'
validate_dir = '/home/minhpn/Desktop/Green_Parking/Test_models/YOLO_CNN/Final_Result/ocr_results.txt'
compare_txt_files(test_dir, validate_dir)