import os
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression, scale_boxes
import cv2
import torch
from ultralytics.data.augment import LetterBox
# import torchvision.transforms as T
import numpy as np

class LetterExtractor:
    def __init__(self, data_dir, save_dir, best_model_file, debug_mode=False):
        self.model_file = best_model_file
        self.model = YOLO(self.model_file)
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.debug_mode = debug_mode
        os.makedirs(save_dir, exist_ok=True)
        self.names = self.model.model.names

    def process_images(self):
        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith('.jpg'):
                if self.debug_mode:
                    print("Model class names:", self.names)
                self._process_single_image(filename)
        print("Successfully processed images")

    def _process_single_image(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        detections = []

        # Load image
        img0 = cv2.imread(filepath)
        assert img0 is not None, f"Failed to load {filepath}"

        # Resize and pad image using letterbox
        img_resized = LetterBox(img0, new_shape=(640, 640))[0]

        # Convert BGR to RGB, transpose to CHW, normalize to [0,1]
        img = img_resized[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0).to(self.model.device)

        # Run model
        with torch.no_grad():
            pred = self.model.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

        if pred is None or len(pred) == 0:
            print(f"No characters detected in {filename}")
            output_path = os.path.join(self.save_dir, 'ocr_results.txt')
            with open(output_path, 'a') as f:
                f.write(f"{filename[:12]}.jpg: None\n")
            return

        # Scale boxes to original image
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], img0.shape).round()

        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = xyxy
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            class_name = self.names[int(cls)]
            detections.append((x_center.item(), y_center.item(), class_name))

        y_values = [d[1] for d in detections]
        y_median = sorted(y_values)[len(y_values) // 2]
        line1 = [det for det in detections if det[1] < y_median]
        line2 = [det for det in detections if det[1] >= y_median]

        line1 = sorted(line1, key=lambda x: x[0])
        line2 = sorted(line2, key=lambda x: x[0])
        final_characters = [char for _, _, char in line1 + line2]
        predicted_text = ''.join(final_characters)
        predicted_text_process = predicted_text[:2] + '-' + predicted_text[2:4] + ' ' + predicted_text[4:]

        if self.debug_mode:
            print(f"{filename}: Raw Prediction = {predicted_text}")
            print(f"{filename}: Formatted Plate = {predicted_text_process}")

        output_path = os.path.join(self.save_dir, 'ocr_results.txt')
        with open(output_path, 'a') as f:
            f.write(f'{filename[:12]}.jpg: {predicted_text_process}\n')
            print(f"Finished processing {filename[:12]}")



