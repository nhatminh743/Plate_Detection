model_path = r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_Detect_Number_From_Plate/runs/detect/train5/weights/best.pt'

image_path = (r'/home/minhpn/Desktop/Green_Parking/Hung_0439_png.rf.f657486e2ef1dfed77c819713baed990.jpg'r'')

import cv2
from ultralytics import YOLO

model = YOLO(model_path)

sample_img = cv2.imread(image_path)

sample_img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

result = model.predict(sample_img_rgb)[0]

# Access detections
boxes = result.boxes  # contains xyxy, confidence, class

for box in boxes:
    # xyxy is (x1, y1, x2, y2)
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    confidence = box.conf[0].item()
    cls = box.cls[0].item()

    print(f"Plate at ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f}), Confidence: {confidence:.2f}, Class: {cls}")

