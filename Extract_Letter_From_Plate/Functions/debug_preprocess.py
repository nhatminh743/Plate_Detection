import cv2
import extract_plate_function

image = cv2.imread('/home/minhpn/Desktop/Green_Parking/Small_Dummy_Data/Raw_Data/0229_05817_b.jpg')

img, imgThresh, status = extract_plate_function.detect_license_plate(image, imshow_mode=True)

