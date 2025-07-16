import os
import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image

model = YOLO('/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_Detect_Number_From_Plate/runs/content/runs/detect/train2/weights/best.pt')

image_path = "/home/minhpn/Desktop/Green_Parking/Dummy_Data_For_Small_Test/Extracted_Plate_Data/0229_05817_b_plate.jpg"
output_dir = '/home/minhpn/Desktop/Green_Parking/Dummy_Data_For_Small_Test/Final_Result'
os.makedirs(output_dir, exist_ok=True)

# Load image
original_image = Image.open(image_path)
scale_factor = 6  # or any number you like

# Resize image before detection
new_size = (original_image.width * scale_factor, original_image.height * scale_factor)
resized_image = original_image.resize(new_size, Image.BICUBIC)

# Run detection on resized image
results = model.predict(source=resized_image, save=False, imgsz=new_size)[0]

# Plot results
img_with_boxes = results.plot(font_size=scale_factor * 10, pil=True, line_width=scale_factor * 2)

# Save image
output_path = os.path.join(output_dir, os.path.basename(image_path))
img_with_boxes.save(output_path)
