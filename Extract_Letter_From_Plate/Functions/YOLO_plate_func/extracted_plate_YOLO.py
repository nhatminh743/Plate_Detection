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
        self.fail_count = 0  # ✅ Add this
        self.total_images = 0
        self.all_image_dimension = []
        self.org_dim = []
        self.model_dim = []

    def process_images(self):
        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith('.jpg'):
                dimension, org_dim, model_dim = self._process_single_image(filename)
                if dimension:
                    self.all_image_dimension.append(dimension)
                    self.org_dim.append(org_dim)
                    self.model_dim.append(model_dim)
        self._report()
        return self.all_image_dimension, self.org_dim, self.model_dim

    def _process_single_image(self, filename):
        file_path = os.path.join(self.data_dir, filename)
        img = cv2.imread(file_path)

        if img is None:
            print(f"❌ Failed to read image: {filename}")
            self.fail_count += 1
            return None, None, 1

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.model.predict(rgb_img)[0]

        model_h, model_w = result.orig_shape[:2]
        orig_h, orig_w = img.shape[:2]

        scale_x = orig_w / model_w
        scale_y = orig_h / model_h

        if result.boxes is None or len(result.boxes) < 1:
            print(f"❌ No plate detected: {filename}")
            return None, None, 2

        #Save dimension
        for box in result.boxes.xyxy.tolist():
            x1, y1, x2, y2 = box

        x1 *= scale_x
        y1 *= scale_x
        x2 *= scale_y
        y2 *= scale_y

        dimension = [x1, y1, x2, y2]

        # Save cropped region & calculate failure score
        not_pass_confidence, not_pass_ratio = crop_and_save_rois(rgb_img, result, self.save_dir, filename)

        if (not_pass_ratio + not_pass_confidence) >= 1:
            self.fail_confidence += not_pass_confidence
            self.fail_ratio += not_pass_ratio

        self.total_images += 1

        return dimension, [orig_w, orig_h], [model_w, model_h]

    def _report(self):
        print("\n" + "=" * 50)
        print("SUMMARY REPORT")
        print("=" * 50)
        print(f"Total Images Processed   : {self.total_images}")
        print(f"Images Failed to Load    : {self.fail_count}")
        print(f"Failed Ratio (Total)     : {self.fail_ratio}")
        print(f"Fail Confidence (Total)  : {self.fail_confidence}")
        print("=" * 50 + "\n")
