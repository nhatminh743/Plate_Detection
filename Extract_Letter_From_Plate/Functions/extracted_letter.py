from extracted_letter_function import extract_letter_from_plate
import cv2
import os

data_dir = r'/home/minhpn/Desktop/Green_Parking/Small_Dummy_Data/Extracted_Plate'
save_dir = r'/home/minhpn/Desktop/Green_Parking/Small_Dummy_Data/Extracted_Letter'
#
# for filename in os.listdir(data_dir):
#     if filename.lower().endswith('.jpg'):
#         file_path = os.path.join(data_dir, filename)
#
#         nothing = extract_letter_from_plate(save_dir, file_path, imshow=True)
#         break

file_path = '/home/minhpn/Desktop/Green_Parking/Small_Dummy_Data/Extracted_Plate/0229_05817_b_plate.jpg'

nothing = extract_letter_from_plate(save_dir, file_path, imshow=True)
