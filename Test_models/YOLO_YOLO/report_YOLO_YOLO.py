import sys
sys.path.append('/home/minhpn/Desktop/Green_Parking/Test_models')
from Test_models.report_function import compare_txt_files

def report_result_YOLO_YOLO(long_report=True):
    test_dir = '/home/minhpn/Desktop/Green_Parking/Test_models/Validation.txt'
    validate_dir = '/home/minhpn/Desktop/Green_Parking/Test_models/YOLO_YOLO/Final_Result/ocr_results.txt'
    list_of_mismatch = compare_txt_files(test_dir, validate_dir, long_report=True, saved=True)
    print(list_of_mismatch)


if __name__ == '__main__':
    report_result_YOLO_YOLO()