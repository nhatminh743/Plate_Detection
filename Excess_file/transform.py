import os

# Base directory
base_dir = "/Use_full_model/data"

# Folders to check
label_dirs = ["train/labels", "validation/labels"]

# Target class ID for LPd
target_class_id = '1'

# Files that contain LPd
files_with_lpd = []

for subdir in label_dirs:
    label_path = os.path.join(base_dir, subdir)
    for filename in os.listdir(label_path):
        if filename.endswith(".txt"):
            full_path = os.path.join(label_path, filename)
            with open(full_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip().startswith(target_class_id + ' '):
                        files_with_lpd.append(full_path)
                        break

# Output result
print(f"Files containing LPd (class ID = {target_class_id}):")
for path in files_with_lpd:
    print(path)
