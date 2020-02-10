import glob
import os

"""
Use this script to find the big images (6000x8000) in a folder. Useful for preserving the same train/validation/test 
split across different image filtering/pre-processing methods. A big image is identified by the first 15 characters in
the filename. 
"""

if __name__ == '__main__':
    FOLDER = r"/media/kitkat/Seagate Expansion Drive/Master_project/machine_learning_dataset_3/train/images"
    paths = glob.glob(os.path.join(FOLDER, "*.tif"))
    paths = [os.path.split(path)[-1] for path in paths]
    paths = [path[:15] for path in paths]
    paths = list(set(paths))  # Remove duplicates
    paths.sort()

    for path in paths:
        print(path)
