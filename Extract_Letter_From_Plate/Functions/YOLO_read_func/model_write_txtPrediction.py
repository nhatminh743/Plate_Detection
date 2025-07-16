# Create a YOLOv8 model
from ultralytics import YOLO

model = YOLO('/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_Detect_Number_From_Plate/runs/content/runs/detect/train2/weights/best.pt')

# Read the image and perform object detection on it
image_path = "/home/minhpn/Desktop/Green_Parking/Dummy_Data_For_Small_Test/Extracted_Plate_Data/0228_01392_b_plate.jpg"
info_path = r'/home/minhpn/Desktop/Green_Parking/Dummy_Data_For_Small_Test/Final_Result/ocr_results.txt'
res = model.predict(image_path)[0]

names = model.model.names

for box in res.boxes:
    cls_id = int(box.cls[0])
    class_name = names[cls_id]
    x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
    y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
    detections.append((x_center.item(), y_center.item(), class_name))