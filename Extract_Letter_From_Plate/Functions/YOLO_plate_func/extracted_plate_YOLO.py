import os
import sys
import cv2
sys.path.append('/Extract_Letter_From_Plate/Full_pipeline_YOLO_EasyOCR')
from Extract_Letter_From_Plate.Functions.YOLO_plate_func.plotting import plot_result, crop_and_save_rois
from ultralytics import YOLO


class PlateExtractor:
    def __init__(self, data_dir, save_dir, best_model_file, debug_mode=False):
        self.model_file = best_model_file
        self.model = YOLO(self.model_file)
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.debug_mode = debug_mode
        os.makedirs(save_dir, exist_ok=True)
        self.fail_confidence = 0
        self.fail_ratio = 0
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
        not_pass_confidence, not_pass_ratio = crop_and_save_rois(rgb_img, result, self.save_dir, filename)

        if (not_pass_ratio + not_pass_confidence) >= 1:
            self.fail_confidence += not_pass_confidence
            self.fail_ratio += not_pass_ratio

        self.total_images += 1

    def _report(self):
        print("\n" + "=" * 50)
        print("SUMMARY REPORT")
        print("=" * 50)
        print(f"Total Images Processed : {self.total_images}")
        print(f"Failed Ratio      : {self.fail_ratio}")
        print(f"Fail Confidence   : {self.fail_confidence}")
        print("Note that fail confidence and fail ratio don't matter to the successful rate.")
        print("=" * 50 + "\n")
