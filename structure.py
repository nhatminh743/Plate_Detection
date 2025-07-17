import os

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            if f.endswith('.jpg') or f.endswith('.png'):
                continue
            print('{}{}'.format(subindent, f))

list_files(r'/home/minhpn/Desktop/Green_Parking')