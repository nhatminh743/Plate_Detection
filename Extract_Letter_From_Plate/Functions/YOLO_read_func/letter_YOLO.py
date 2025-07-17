import os
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression, scale_boxes
import cv2
import torch
from ultralytics.data.augment import LetterBox
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
        self.scale_factor = 6

    def process_images(self):
        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith('.jpg'):
                self._process_single_image(filename)
        print("Successfully processed images")

    def _process_single_image(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        detections = []

        # Load image
        original_image = cv2.imread(filepath)
        assert original_image is not None, f"Failed to load {filepath}"
        h, w, channel = original_image.shape

        # Resize image
        new_size = (w * self.scale_factor, h * self.scale_factor)
        resized_image = cv2.resize(original_image, new_size, interpolation=cv2.INTER_LINEAR)

        # Run model inference using Ultralytics high-level API
        results = self.model.predict(resized_image, imgsz=new_size, conf=0.25, iou=0.7, agnostic_nms = True)[0]

        # If no boxes found
        if results.boxes is None or len(results.boxes) == 0:
            print(f"No characters detected in {filename}")
            output_path = os.path.join(self.save_dir, 'ocr_results.txt')
            with open(output_path, 'a') as f:
                f.write(f"{filename[:12]}.jpg: None\n")
            return

        # Process each detected box
        for box in results.boxes:
            x, y, w, h = box.xywh[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = self.model.names[cls]
            print(f"Box: ({x:.0f}, {y:.0f}, {w:.0f}, {h:.0f}), conf: {conf:.2f}, class: {class_name}")

            # Compute center for sorting
            x_center = x
            y_center = y
            detections.append((x_center, y_center, class_name))

        # Organize characters into 2 lines based on y-median
        y_values = [d[1] for d in detections]
        sum_y = sum(y_values)
        y_mean = sum_y / len(y_values)
        line1 = [det for det in detections if det[1] < y_mean]
        line2 = [det for det in detections if det[1] >= y_mean]

        # Sort characters left to right
        line1 = sorted(line1, key=lambda x: x[0])
        line2 = sorted(line2, key=lambda x: x[0])
        final_characters = [char for _, _, char in line1 + line2]
        predicted_text = ''.join(final_characters)

        # Format plate: XX-YY ZZZZ (or whatever format you want)
        predicted_text_process = predicted_text[:2] + '-' + predicted_text[2:4] + ' ' + predicted_text[4:]

        # Debug print
        if self.debug_mode:
            print(f"{filename}: Raw Prediction = {predicted_text}")
            print(f"{filename}: Formatted Plate = {predicted_text_process}")

        # Save result
        output_path = os.path.join(self.save_dir, 'ocr_results.txt')
        with open(output_path, 'a') as f:
            f.write(f'{filename[:12]}.jpg: {predicted_text_process}\n')
            print(f"Finished processing {filename[:12]}")

    # def _process_single_image(self, filename):
    #     filepath = os.path.join(self.data_dir, filename)
    #     detections = []
    #
    #     # Load image
    #     img0 = cv2.imread(filepath)
    #     assert img0 is not None, f"Failed to load {filepath}"
    #
    #     # Run model inference using Ultralytics high-level API
    #     results = self.model.predict(img0, imgsz=640, conf=0.5, iou=0.65)[0]
    #
    #     # If no boxes found
    #     if results.boxes is None or len(results.boxes) == 0:
    #         print(f"No characters detected in {filename}")
    #         output_path = os.path.join(self.save_dir, 'ocr_results.txt')
    #         with open(output_path, 'a') as f:
    #             f.write(f"{filename[:12]}.jpg: None\n")
    #         return
    #
    #     # Process each detected box
    #     for box in results.boxes:
    #         x1, y1, x2, y2 = box.xyxy[0].tolist()
    #         conf = float(box.conf[0])
    #         cls = int(box.cls[0])
    #         class_name = self.model.names[cls]
    #         print(f"Box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}), conf: {conf:.2f}, class: {class_name}")
    #
    #         # Compute center for sorting
    #         x_center = (x1 + x2) / 2
    #         y_center = (y1 + y2) / 2
    #         detections.append((x_center, y_center, class_name))
    #
    #     # Organize characters into 2 lines based on y-median
    #     y_values = [d[1] for d in detections]
    #     y_median = sorted(y_values)[len(y_values) // 2]
    #     line1 = [det for det in detections if det[1] < y_median]
    #     line2 = [det for det in detections if det[1] >= y_median]
    #
    #     # Sort characters left to right
    #     line1 = sorted(line1, key=lambda x: x[0])
    #     line2 = sorted(line2, key=lambda x: x[0])
    #     final_characters = [char for _, _, char in line1 + line2]
    #     predicted_text = ''.join(final_characters)
    #
    #     # Format plate: XX-YY ZZZZ (or whatever format you want)
    #     predicted_text_process = predicted_text[:2] + '-' + predicted_text[2:4] + ' ' + predicted_text[4:]
    #
    #     # Debug print
    #     if self.debug_mode:
    #         print(f"{filename}: Raw Prediction = {predicted_text}")
    #         print(f"{filename}: Formatted Plate = {predicted_text_process}")
    #
    #     # Save result
    #     output_path = os.path.join(self.save_dir, 'ocr_results.txt')
    #     with open(output_path, 'a') as f:
    #         f.write(f'{filename[:12]}.jpg: {predicted_text_process}\n')
    #         print(f"Finished processing {filename[:12]}")

    # def _process_single_image(self, filename):
    #     filepath = os.path.join(self.data_dir, filename)
    #     detections = []
    #
    #     # Load image
    #     img0 = cv2.imread(filepath)
    #     assert img0 is not None, f"Failed to load {filepath}"
    #
    #     # Resize and pad image using letterbox
    #     transformer = LetterBox(new_shape=(640, 640))
    #     img_resized = transformer(image=img0)
    #
    #     # Convert BGR to RGB, transpose to CHW, normalize to [0,1]
    #     img = img_resized[:, :, ::-1].transpose(2, 0, 1)
    #     img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    #     img = torch.from_numpy(img).unsqueeze(0).to(self.model.device)
    #
    #     # Run model
    #     with torch.no_grad():
    #         pred = self.model.model(img)[0]
    #
    #     # Apply NMS
    #     pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.65)[0]
    #
    #     if pred is not None:
    #         for box in pred:
    #             x1, y1, x2, y2, conf, cls = box
    #             print(f"Box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}), conf: {conf:.2f}, class: {self.model.names[int(cls)]}")
    #     else:
    #         print("No detections after NMS.")
    #
    #     results = self.model.predict(img0, imgsz=640, conf=0.5, iou=0.65)[0]
    #     for box in results.boxes:
    #         x1, y1, x2, y2 = box.xyxy[0].tolist()
    #         conf = float(box.conf[0])
    #         cls = int(box.cls[0])
    #         print(f"Box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}), conf: {conf:.2f}, class: {self.model.names[cls]}")
    #
    #     if pred is None or len(pred) == 0:
    #         print(f"No characters detected in {filename}")
    #         output_path = os.path.join(self.save_dir, 'ocr_results.txt')
    #         with open(output_path, 'a') as f:
    #             f.write(f"{filename[:12]}.jpg: None\n")
    #         return
    #
    #     # Scale boxes to original image
    #     pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], img0.shape).round()
    #
    #     for *xyxy, conf, cls in pred:
    #         x1, y1, x2, y2 = xyxy
    #         x_center = (x1 + x2) / 2
    #         y_center = (y1 + y2) / 2
    #         class_name = self.names[int(cls)]
    #         detections.append((x_center.item(), y_center.item(), class_name))
    #
    #     y_values = [d[1] for d in detections]
    #     y_median = sorted(y_values)[len(y_values) // 2]
    #     line1 = [det for det in detections if det[1] < y_median]
    #     line2 = [det for det in detections if det[1] >= y_median]
    #
    #     line1 = sorted(line1, key=lambda x: x[0])
    #     line2 = sorted(line2, key=lambda x: x[0])
    #     final_characters = [char for _, _, char in line1 + line2]
    #     predicted_text = ''.join(final_characters)
    #     predicted_text_process = predicted_text[:2] + '-' + predicted_text[2:4] + ' ' + predicted_text[4:]
    #
    #     if self.debug_mode:
    #         print(f"{filename}: Raw Prediction = {predicted_text}")
    #         print(f"{filename}: Formatted Plate = {predicted_text_process}")
    #
    #     output_path = os.path.join(self.save_dir, 'ocr_results.txt')
    #     with open(output_path, 'a') as f:
    #         f.write(f'{filename[:12]}.jpg: {predicted_text_process}\n')
    #         print(f"Finished processing {filename[:12]}")


