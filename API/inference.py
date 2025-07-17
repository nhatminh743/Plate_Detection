# inference.py
import numpy as np
import cv2
from model_loader import load_model

detect_plate_model = r'/home/minhpn/Desktop/Green_Parking/Model_training/YOLOv11_training/runs/detect/train3/weights/best.pt'

model = load_model(detect_plate_model)

def run_yolo_inference(image: np.ndarray, conf: float = 0.25):
    results = model.predict(image, imgsz=640, conf=conf)
    detections = []

    # Parse detection results
    for box in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, cls_id = box
        detections.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(score),
            "class_id": int(cls_id)
        })
    return detections
