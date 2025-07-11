import sys
sys.path.append('/home/minhpn/Desktop/Green_Parking/Test')

from report import compare_txt_files

test_dir = '/home/minhpn/Desktop/Green_Parking/Test/Validation.txt'
validate_dir = '/home/minhpn/Desktop/Green_Parking/Test/YOLO_EasyOCR/Final_Result/ocr_results.txt'
compare_txt_files(test_dir, validate_dir)