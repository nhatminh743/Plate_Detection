from YOLO_EasyOCR import report_result_YOLO_EasyOCR
from YOLO_CNN import report_result_YOLO_CNN
from OpenCV_EasyOCR import report_result_OpenCV_EasyOCR
from OpenCV_CNN import report_result_OpenCV_CNN
from sort_alphabetically_txt import sort_txt_by_title

def report_all():
    sort_all()
    report_individual()

def sort_all():
    sort_txt_by_title('/home/minhpn/Desktop/Green_Parking/Test_models/YOLO_EasyOCR/Final_Result/ocr_results.txt')
    sort_txt_by_title('/home/minhpn/Desktop/Green_Parking/Test_models/YOLO_CNN/Final_Result/ocr_results.txt')
    sort_txt_by_title('/home/minhpn/Desktop/Green_Parking/Test_models/OpenCV_EasyOCR/Final_Result/ocr_results.txt')
    sort_txt_by_title('/home/minhpn/Desktop/Green_Parking/Test_models/OpenCV_CNN/Final_Result/ocr_results.txt')

def report_individual():
    reports = [
        ('YOLO_EasyOCR', report_result_YOLO_EasyOCR),
        ('YOLO_CNN', report_result_YOLO_CNN),
        ('OpenCV_EasyOCR', report_result_OpenCV_EasyOCR),
        ('OpenCV_CNN', report_result_OpenCV_CNN),
    ]

    for name, report_func in reports:
        print('=' * 100)
        print(f"Running report for: {name}")
        report_func(long_report=False)
        print('=' * 100)

if __name__ == '__main__':
    report_all()
