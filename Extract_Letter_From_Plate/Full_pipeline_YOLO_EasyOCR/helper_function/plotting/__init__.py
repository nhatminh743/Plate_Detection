from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import os
import cv2
import numpy as np

def convert_to_pixel_coords(box, img_width, img_height):
    """
    Convert YOLO format box (cx, cy, w, h) to pixel coordinates.
    """
    return [
        box[0] * img_width,  # cx
        box[1] * img_height, # cy
        box[2] * img_width,  # w
        box[3] * img_height  # h
    ]

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    Boxes are in the format (cx, cy, w, h).
    """
    # Convert (cx, cy, w, h) to (x1, y1, x2, y2)
    x1_box1, y1_box1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x2_box1, y2_box1 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2

    x1_box2, y1_box2 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_box2, y2_box2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    # Calculate intersection
    inter_x1 = max(x1_box1, x1_box2)
    inter_y1 = max(y1_box1, y1_box2)
    inter_x2 = min(x2_box1, x2_box2)
    inter_y2 = min(y2_box1, y2_box2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate union
    box1_area = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
    box2_area = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def plot_result(rgb, result, label=None, iou_threshold=0.5):
    """
    Plot YOLO prediction results and compare with ground truth labels if provided.

    Parameters:
      - rgb: numpy array of the RGB image.
      - result: YOLO prediction result.
      - label: Optional ground truth labels [(class_id, [cx, cy, w, h]), ...].
      - iou_threshold: IoU threshold to match predictions with ground truth.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb)

    class_names = result.names
    height, width, _ = rgb.shape

    predictions = result.boxes.xywh.cpu().numpy()
    pred_classes = result.boxes.cls.cpu().numpy()

    gt_used = [False] * len(label) if label else []

    for i, pred_box in enumerate(predictions):
        pred_class_id = int(pred_classes[i])
        pred_box = pred_box.tolist()
        matched = False

        if label:
            for j, (gt_class_id, gt_box) in enumerate(label):
                if not gt_used[j] and gt_class_id == pred_class_id:
                    # Convert ground truth box to pixel values
                    gt_box_pixel = convert_to_pixel_coords(gt_box, width, height)
                    iou = calculate_iou(pred_box, gt_box_pixel)
                    if iou >= iou_threshold:
                        matched = True
                        gt_used[j] = True
                        break

        color = 'green' if matched else 'red'
        cx, cy, w, h = pred_box
        hw, hh = w / 2, h / 2

        ax.add_patch(Rectangle(
            (cx - hw, cy - hh), w, h,
            edgecolor=color,
            fill=None,
            linewidth=2
        ))

        label_text = f"{class_names[pred_class_id]} ({iou:.2f})" if matched else class_names[pred_class_id]
        ax.text(
            cx - hw, cy - hh - 5,
            label_text,
            color=color,
            fontsize=10,
            fontweight='bold',
            bbox=dict(facecolor='white', edgecolor=color, alpha=0.7)
        )

    if label:
        for gt_class_id, gt_box in label:
            # Convert ground truth box to pixel values
            gt_box_pixel = convert_to_pixel_coords(gt_box, width, height)
            cx, cy, w, h = gt_box_pixel
            hw, hh = w / 2, h / 2

            ax.add_patch(Rectangle(
                (cx - hw, cy - hh), w, h,
                edgecolor='blue',
                fill=None,
                linestyle='--',
                linewidth=1
            ))

    plt.show()

def crop_and_save_rois(rgb, result, save_dir, filename, conf_threshold=0.5):
    """
    Crop ROIs from YOLO predictions and save them to a folder.

    Parameters:
      - rgb: numpy array of the RGB image.
      - result: YOLO prediction result.
      - save_dir: directory to save cropped images.
      - conf_threshold: confidence threshold to filter predictions.
    """
    os.makedirs(save_dir, exist_ok=True)
    height, width, _ = rgb.shape

    boxes = result.boxes.xywh.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    fail_count = 0

    for i, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
        if score < conf_threshold:
            fail_count =1
            continue

        cx, cy, w, h = box
        x1 = int(max(cx - w / 2, 0))
        y1 = int(max(cy - h / 2, 0))
        x2 = int(min(cx + w / 2, width))
        y2 = int(min(cy + h / 2, height))

        roi = rgb[y1:y2, x1:x2]

        save_path = os.path.join(save_dir, f"{filename[:12]}_plate.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
        print(f"Saved ROI to {save_path}")

    return fail_count