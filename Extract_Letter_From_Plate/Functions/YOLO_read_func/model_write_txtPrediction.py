import os
import numpy as np
from ultralytics import YOLO
import cv2

model = YOLO('/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_Detect_Number_From_Plate/runs/content/runs/detect/train2/weights/best.pt')

image_dir = "/home/minhpn/Desktop/Green_Parking/Dummy_Data_For_Small_Test/Extracted_Plate_Data/0229_05817_b_plate.jpg"
output_dir = '/home/minhpn/Desktop/Green_Parking/Dummy_Data_For_Small_Test/Final_Result'

os.makedirs(output_dir, exist_ok=True)

results = model.predict(image_dir)[0]

# Plot results and get PIL image
img = results.plot(font_size=10, pil=True, line_width=2)

# Convert PIL image to OpenCV-compatible format
img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Save the image
cv2.imwrite(os.path.join(output_dir, os.path.basename(image_dir)), img_cv)

