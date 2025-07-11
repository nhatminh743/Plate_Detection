from ultralytics import YOLO
import cv2
from plotting import plot_result, crop_and_save_rois

model_file = '/home/minhpn/Desktop/Green_Parking/Use_full_model/runs/detect/train2/weights/best.pt'
model = YOLO(model_file)

image = cv2.imread('/home/minhpn/Desktop/Green_Parking/Small_Dummy_Data/Raw_Data/0228_01938_b.jpg')
save_dir = '/home/minhpn/Desktop/Green_Parking/Small_Dummy_Data/testing'

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
filename = '0228_01938_b.jpg'

results = model.predict(rgb, conf = 0.5, iou = 0.3)[0]

plot_result(rgb, results)

crop_and_save_rois(rgb, results, save_dir, filename)