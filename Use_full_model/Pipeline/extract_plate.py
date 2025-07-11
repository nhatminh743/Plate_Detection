import os
import cv2
import numpy as np
from Extract_Letter_From_Plate.Functions import extract_plate_function
from plotting import plot_result, crop_and_save_rois
from ultralytics import YOLO

class PlateExtractor:
    def __init__(self, data_dir, save_dir, best_model_file, debug_mode=False):
        self.model_file = best_model_file
        self.model = YOLO(self.model_file)
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.debug_mode = debug_mode
        os.makedirs(save_dir, exist_ok=True)
        self.fail_count = 0
        self.total_images = 0

    def process_images(self):
        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith('.jpg'):
                self._process_single_image(filename)
        self._report()

    def _process_single_image(self, filename):
        file_path = os.path.join(self.data_dir, filename)
        img = cv2.imread(file_path)

        if img is None:
            print(f"Failed to read image: {filename}")
            self.fail_count += 1
            return

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.model.predict(rgb_img)[0]

        # Crop & Save ROIs, passing filename for organized saving
        fail_status = crop_and_save_rois(rgb_img, result, self.save_dir, filename)

        if fail_status:
            self.fail_count += 1

        self.total_images += 1

    def _report(self):
        print("\n" + "=" * 50)
        print("SUMMARY REPORT")
        print("=" * 50)
        print(f"Total Images Processed : {self.total_images}")
        print(f"Failed Detections       : {self.fail_count}")
        success = self.total_images - self.fail_count
        print(f"Successful Detections   : {success}")
        print("=" * 50 + "\n")
