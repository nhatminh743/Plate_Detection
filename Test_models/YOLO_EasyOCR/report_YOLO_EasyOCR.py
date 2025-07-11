import sys
sys.path.append('/Test_models')

from report import compare_txt_files

test_dir = '/Test_models/Validation.txt'
validate_dir = '/Test_models/YOLO_EasyOCR/Final_Result/ocr_results.txt'
compare_txt_files(test_dir, validate_dir)