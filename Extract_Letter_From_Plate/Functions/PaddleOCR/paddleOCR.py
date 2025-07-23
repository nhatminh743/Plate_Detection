import os
import cv2
import numpy as np
from paddleocr import TextDetection, TextRecognition

class PaddleOCRLineExtractor:
    def __init__(self, image_path, save_dir):
        self.image_path = image_path
        self.save_dir = save_dir
        self.text_detector = TextDetection(model_name="PP-OCRv5_server_det")
        self.text_recognizer = TextRecognition()
        os.makedirs(self.save_dir, exist_ok=True)

    def detect_and_crop_lines(self):
        image = cv2.imread(self.image_path)
        output = self.text_detector.predict(self.image_path, batch_size=1)

        for res in output:
            sort_based_on_y_coor = sorted(res['dt_polys'], key=lambda coor: np.mean(coor[:, 1]))
            for i, poly in enumerate(sort_based_on_y_coor):
                x_coords = poly[:, 0]
                y_coords = poly[:, 1]

                x1, x2 = int(np.min(x_coords)), int(np.max(x_coords))
                y1, y2 = int(np.min(y_coords)), int(np.max(y_coords))

                y_mean = np.mean(y_coords)

                roi = image[y1:y2, x1:x2]

                if roi.size == 0:
                    print(f"❌ Empty ROI at index {i}, skipping.")
                    continue

                roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
                save_path = os.path.join(self.save_dir, f"roi_{i}.jpg")
                cv2.imwrite(save_path, roi_bgr)
                print(f"Saved ROI to {save_path}")

    def recognize_text_from_lines(self):
        all_texts = []

        for filename in sorted(os.listdir(self.save_dir)):
            filepath = os.path.join(self.save_dir, filename)
            output = self.text_recognizer.predict(input=filepath)
            for res in output:
                all_texts.append(res['rec_text'])

        final_line = ' '.join(all_texts)
        print(final_line)

    def run(self):
        print("Detecting and cropping lines...")
        self.detect_and_crop_lines()
        print("Recognizing text from cropped lines...")
        self.recognize_text_from_lines()

new = PaddleOCRLineExtractor(image_path = "/home/minhpn/Desktop/Green_Parking/one_image/Extracted_Plate_Data/bien-so-xe-phong-thuy-6_plate_0.jpg"
,save_dir = "/home/minhpn/Desktop/Green_Parking/one_image/Extracted_Line")

new.run()