import os
import json
from PIL import Image

classes_path = '/home/minhpn/Desktop/Green_Parking/Model_training/Temp/validate/labels/0112_01913_b_plate.txt'
output_dir = '/home/minhpn/Desktop/Green_Parking/Model_training/Temp/validate/new_file.json'

with open(classes_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

os.makedirs(output_dir, exist_ok=True)

for label_file in os.listdir(classes_path):
    if not label_file.endswith(".txt"):
        continue

    file_name = label_file.replace(".txt", ".json")
    file_path = os.path.join(output_dir, file_name)
    label_path = os.path.join(classes_path, label_file)

    if not os.path.exists(file_path):
        print(f"Image not found for {label_file}, skipping.")
        continue

    # Prepare structure
    data = {

    }

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, x_center, y_center, width, height = map(float, parts)
            cls_id = int(cls_id)
            if cls_id >= len(class_names):
                print(f"Class ID {cls_id} out of range in {label_file}, skipping.")
                continue
            label = class_names[cls_id]

            # Convert YOLO to absolute box coordinates
            x = x_center * w
            y = y_center * h
            box_w = width * w
            box_h = height * h
            x1 = x - box_w / 2
            y1 = y - box_h / 2
            x2 = x + box_w / 2
            y2 = y + box_h / 2

            # Add rectangle shape
            shape = {
                "label": label,
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            data["shapes"].append(shape)

    # Save to JSON
    output_path = os.path.join(output_dir, label_file.replace(".txt", ".json"))
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

