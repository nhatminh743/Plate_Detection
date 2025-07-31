import os
import shutil


def copy_all_files(source_dir: str, target_dir: str):
    """
    Copies all files from source_dir to target_dir.

    Args:
        source_dir (str): Path to the directory containing files to copy.
        target_dir (str): Path to the destination directory.
    """
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(target_dir, filename)

        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)  # copy2 keeps metadata (timestamps, etc.)

    print('Copied all files from source_dir to target_dir')

copy_all_files(
    source_dir=r'/home/minhpn/Desktop/Green_Parking/Model_training/Text_Recognition_Data_Gen/plate_image_gen/divide/white',
    target_dir=r'/Model_training/PaddleOCR_finetune/new_data/label'
)