import os

# File paths
input_txt = ("/home/minhpn/Desktop/Green_Parking/Model_training/PaddleOCR_finetune/data/copy.txt")
output_txt = "/home/minhpn/Desktop/Green_Parking/Model_training/PaddleOCR_finetune/data/label/final_res.txt"

# Set the target directory to replace with
new_subdir = "/label/"

with open(input_txt, "r") as fin, open(output_txt, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            full_path, label = line.split("\t")
            filename = os.path.basename(full_path)  # e.g., CarLongPlateGen..._roi_0_0.jpg

            # Replace only up to /extracted_line/.../
            root_dir = "/home/minhpn/Desktop/Green_Parking/Model_training/PaddleOCR_finetune/data"
            new_path = os.path.join(root_dir, "label", filename)

            fout.write(f"{new_path}\t{label}\n")
        except ValueError:
            print(f"⚠️ Skipping malformed line: {line}")
            continue

print(f"✅ Rewritten paths saved to: {output_txt}")
