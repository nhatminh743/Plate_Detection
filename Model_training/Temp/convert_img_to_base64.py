import json
import base64
import os
from PIL import Image  # ✅ Use Pillow, not tkinter

image_dir = '/home/minhpn/Desktop/Green_Parking/Model_training/Temp/validate/images'
json_dir = '/Model_training/Temp/labelme_output'

def convert_base64(image_path):
    with open(image_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return encoded

def height_width(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    print('width:', width, 'height:', height)
    return height, width

def update_json_file(image_b64, json_path):
    with open(json_path, "r") as json_file:
        data = json.load(json_file)

    data["imageData"] = image_b64

    with open(json_path, "w") as json_file:
        json.dump(data, json_file)
    print(f"Updated imageData in {json_path}")

def update_height_width(h, w, json_path):
    with open(json_path, "r") as json_file:
        data = json.load(json_file)

    data['imageHeight'] = h
    data['imageWidth'] = w

    with open(json_path, "w") as json_file:
        json.dump(data, json_file)
    print(f"Updated dimensions in {json_path}")

# def convert_coordinates()

# === Main loop ===
for filename in os.listdir(image_dir):
    if filename.endswith(".txt"):
        print(f"Skipping {filename}")
        continue

    name_without_ext = os.path.splitext(filename)[0]
    json_path = os.path.join(json_dir, name_without_ext + '.json')
    image_path = os.path.join(image_dir, filename)

    if not os.path.exists(json_path):
        print(f"JSON file not found for image: {filename}")
        continue

    h, w = height_width(image_path)

    # image_b64 = convert_base64(image_path)
    # update_json_file(image_b64, json_path)
    update_height_width(h, w, json_path)

