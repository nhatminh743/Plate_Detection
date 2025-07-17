import os

def replace_linux_with_windows_paths(root_dir, linux_dir, windows_dir, linux_to_windows = True):
    # Normalize both paths for accurate matching
    linux_dir = os.path.normpath(linux_dir)
    windows_dir = os.path.normpath(windows_dir)

    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            #Scan all python file
            if filename.endswith('.py'):
                file_path = os.path.join(foldername, filename)

                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Replace all occurrences of the linux_dir with windows_dir
                if linux_to_windows:
                    new_content = content.replace(linux_dir, windows_dir)
                else:
                    new_content = content.replace(windows_dir, linux_dir)
                if new_content != content:
                    print(f'Updated: {file_path}')
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)

# Example usage:
replace_linux_with_windows_paths(
    root_dir='C:/Users/ACER/Documents/nhatminh743/Plate_Detection/API',
    linux_dir='/home/minhpn/Desktop/Green_Parking',
    windows_dir='C:/Users/ACER/Documents/nhatminh743/Plate_Detection',
    linux_to_windows=False
)
