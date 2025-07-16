import os
from ultralytics import YOLO


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
                self._process_single_image(filename)
        print("Successfully processed images")

    def _process_single_image(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        detections = []

        res = self.model.predict(filepath)[0]

        for box in res.boxes:
            cls_id = int(box.cls[0])
            class_name = self.names[cls_id]
            x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
            y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
            detections.append((x_center.item(), y_center.item(), class_name))

        if not detections:
            print(f"No characters detected in {filename}")
            return

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
            f.write(f'{filename[:12]}: {predicted_text_process}\n')
            print(f"Finished processing {filename[:12]}")


