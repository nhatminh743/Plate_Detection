import os
import numpy as np
from sklearn.decomposition import PCA
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression, scale_boxes
import cv2
from ultralytics.data.augment import LetterBox
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import silhouette_score
from Extract_Letter_From_Plate.Functions.YOLO_read_func.show_result import PlotImageS
from paddleocr import TextRecognition
from scipy.spatial import Voronoi, voronoi_plot_2d


class LetterExtractor:
    def __init__(self, data_dir, save_dir, best_model_file, debug_mode=False):
        self.model_dir = best_model_file
        self.model = YOLO(self.model_dir)
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.debug_mode = debug_mode
        os.makedirs(save_dir, exist_ok=True)
        self.names = self.model.model.names
        self.scale_factor = 6
        self.two_row = True
        self.paddleOCRmodel = TextRecognition()

    def process_images(self):
        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith('.jpg'):
                self._process_single_image(filename)
        print("Successfully processed images")

    def _process_single_image(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        detections = []
        data_for_kMeans = []
        points = []

        # Load image
        original_image = cv2.imread(filepath)
        assert original_image is not None, f"Failed to load {filepath}"
        h, w, channel = original_image.shape

        # Resize image
        new_size = (w * self.scale_factor, h * self.scale_factor)
        resized_image = cv2.resize(original_image, new_size, interpolation=cv2.INTER_LINEAR)

        results = self.model.predict(resized_image, imgsz=new_size, conf=0.25, iou=0.7, agnostic_nms=True)[0]

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
            if self.debug_mode:
                print(f"Box: ({x:.0f}, {y:.0f}, {w:.0f}, {h:.0f}), conf: {conf:.2f}, class: {class_name}")

            detections.append([x, y, class_name])
            points.append([int(x), int(y)])
            data_for_kMeans.append(y)

        points = np.array(points)
        data_for_kMeans = np.array(data_for_kMeans).reshape(-1, 1)
        y_coords = np.array([row[1] for row in points]).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(y_coords)
        labels = kmeans.labels_

        # Evaluate whether there is one-row or two-row
        score = silhouette_score(data_for_kMeans, labels)

        if score < 0.8:
            self.two_row = False
        else:
            self.two_row = True

        print(f'Silhouette score: {score}')

        if self.debug_mode:

            plot_image_func = PlotImageS(
                model_dir=self.model_dir,
                image_dir=self.data_dir,
                output_dir=r'/home/minhpn/Desktop/Green_Parking/one_image/visualization',
            )

        plot_image_func.plot_all()

        if self.two_row:

            cluster1 = points[labels == 0]
            cluster2 = points[labels == 1]

            if cluster1[:, 1].mean() < cluster2[:, 1].mean():
                top_cluster, bottom_cluster = cluster1, cluster2
            else:
                top_cluster, bottom_cluster = cluster2, cluster1
            # Find the line
            pca = PCA(n_components=1)
            pca.fit(np.vstack([top_cluster, bottom_cluster]))
            direction = pca.components_[0]
            direction = direction / np.linalg.norm(direction)

            top_mean = top_cluster.mean(axis=0)
            bottom_mean = bottom_cluster.mean(axis=0)
            # Find the point it need to passthrough
            mid_point = (top_mean + bottom_mean) / 2

            line_len = 600
            line_vector = direction * line_len / 2
            pt1 = mid_point - line_vector
            pt2 = mid_point + line_vector

            if self.debug_mode:
                plt.figure(figsize=(6, 5))
                plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis')
                plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r--', label='Dividing Line (PCA aligned)')
                plt.title("Clustered Rows with Dividing Line")
                plt.xlabel("Center X")
                plt.ylabel("Center Y")
                plt.legend()
                plt.gca().invert_yaxis()
                plt.grid(True)
                plt.show()

            # Cut the plate


            if self.debug_mode:
                plt.scatter([x for x, y, _ in detections], [y for x, y, _ in detections], c=labels, cmap='viridis')
                plt.gca().invert_yaxis()
                plt.title("Clustering Boxes into Rows")
                plt.xlabel("Center X")
                plt.ylabel("Center Y")
                plt.show()

        else:
            clusters = defaultdict(list)
            for (x, y, cls_name), label in zip(detections, labels):
                clusters[label].append((x, y, cls_name))

            sorted_labels = sorted(detections, key=lambda row: row[0])

            sorted_labels = [row[2] for row in sorted_labels]
            final = ''

            for letter in sorted_labels:
                final += letter
            print("Single line detected")
            final = final[:2] + '-' + final[2:-5] + ' ' + final[-5:][::-1]
            print(f'Sorted labels: {sorted_labels}')
            print(f'Final plate is {final}')

            if self.debug_mode:
                plt.scatter([x for x, y, _ in detections], [y for x, y, _ in detections], c=labels, cmap='viridis')
                plt.gca().invert_yaxis()
                plt.title("Clustering Boxes into Rows")
                plt.xlabel("Center X")
                plt.ylabel("Center Y")
                plt.show()