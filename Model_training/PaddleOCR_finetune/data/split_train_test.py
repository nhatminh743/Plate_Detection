import os
import random
import shutil

# Input files and config
input_txt = r'/home/minhpn/Desktop/Green_Parking/Model_training/PaddleOCR_finetune/data/final.txt'     # your original full annotation file
output_dir = r'/home/minhpn/Desktop/Green_Parking/Model_training/PaddleOCR_finetune/new_data'  # where to save train/test images and txts
test_ratio = 0.2                # 20% test split

# Create directories
train_img_dir = os.path.join(output_dir, "train")
test_img_dir = os.path.join(output_dir, "test")
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(test_img_dir, exist_ok=True)

# Read all lines
with open(input_txt, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

# Shuffle and split
random.shuffle(lines)
split_idx = int(len(lines) * (1 - test_ratio))
train_lines = lines[:split_idx]
test_lines = lines[split_idx:]

# Utility: copy image and update path
def copy_and_rewrite(lines, target_dir, rel_subdir):
    new_lines = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) != 2:
            print(f"⚠️ Skipping malformed line: {line}")
            continue
        img_path, annotation = parts
        img_name = os.path.basename(img_path)
        new_path = os.path.join(target_dir, img_name)

        try:
            shutil.copy(img_path, new_path)
        except Exception as e:
            print(f"❌ Failed to copy {img_path}: {e}")
            continue

        # New line with updated relative path
        new_lines.append(f"{rel_subdir}/{img_name}\t{annotation}\n")
    return new_lines

# Process train/test sets
train_output = copy_and_rewrite(train_lines, train_img_dir, "train")
test_output = copy_and_rewrite(test_lines, test_img_dir, "test")

# Write output txt files
with open(os.path.join(output_dir, "train.txt"), "w") as f:
    f.writelines(train_output)
with open(os.path.join(output_dir, "test.txt"), "w") as f:
    f.writelines(test_output)

print(f"✅ Done! Train: {len(train_output)} | Test: {len(test_output)}")
print(f"📁 Images saved in: {train_img_dir} and {test_img_dir}")
print(f"📝 Annotation files: {output_dir}/train.txt and test.txt")
