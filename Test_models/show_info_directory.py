import os

def count_files(dir_path, extensions):
    return sum(1 for f in os.listdir(dir_path)
               if os.path.isfile(os.path.join(dir_path, f)) and os.path.splitext(f)[1].lower() in extensions)

def count_image_files(dir_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    return count_files(dir_path, image_extensions)

def count_label_files(dir_path):
    label_extensions = {'.txt', '.xml'}
    return count_files(dir_path, label_extensions)

def print_dataset_structure(root_dir, indent="", is_last=True):
    if indent == "":
        print(root_dir)

    items = sorted(os.listdir(root_dir))
    items_count = len(items)

    for i, item in enumerate(items):
        item_path = os.path.join(root_dir, item)
        is_item_last = (i == items_count - 1)

        prefix = "└── " if is_item_last else "├── "
        next_indent = indent + ("    " if is_item_last else "│   ")

        if item == ".DS_Store":
            continue

        if os.path.isdir(item_path):
            # Count raw_image and labels inside this folder
            img_count = count_image_files(item_path)
            lbl_count = count_label_files(item_path)
            counts = []
            if img_count > 0:
                counts.append(f"{img_count} images")
            if lbl_count > 0:
                counts.append(f"{lbl_count} labels")
            counts_text = f" [{', '.join(counts)}]" if counts else ""

            print(f"{indent}{prefix}{item}{counts_text}")
            # Recurse into subfolders
            print_dataset_structure(item_path, indent=next_indent, is_last=is_item_last)

# Example usage:
print_dataset_structure("./")
