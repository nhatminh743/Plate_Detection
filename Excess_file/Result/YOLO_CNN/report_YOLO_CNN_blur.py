import sys
sys.path.append('/Test_models')
from report_function import compare_txt_files

def report_result_YOLO_CNN(long_report = True):
    test_dir = '/Test_models/Validation.txt'
    validate_dir = '/Excess_file/Result/YOLO_CNN/Final_Result/ocr_results.txt'
    compare_txt_files(test_dir, validate_dir, long_report=long_report)

if __name__ == '__main__':
    report_result_YOLO_CNN()
