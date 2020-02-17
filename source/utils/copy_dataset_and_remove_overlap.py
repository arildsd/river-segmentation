import glob
import os
import shutil
import sys

"""
Use this util to copy a dataset while simultaneity removing overlapping images. 
Image names must be in the following format: <big_image_name>_n_<north_offset>_e_<east_offset>.tif
"""
def _copy_and_remove_overlap(source_dir, dest_dir):
    all_images = glob.glob(os.path.join(source_dir, "images", "*.tif"))
    os.makedirs(os.path.join(dest_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "labels"), exist_ok=True)
    for image in all_images:
        splits = os.path.split(image)[-1].split("_")
        north_offset = int(splits[2])
        east_offset = int(splits[4].replace(".tif", ""))
        if north_offset % 512 == 0 and east_offset % 512 == 0:
            shutil.copyfile(image, os.path.join(dest_dir, "images", os.path.split(image)[-1]))
            shutil.copyfile(image.replace("images", "labels"), os.path.join(dest_dir, "labels", os.path.split(image)[-1]))

def run():
    source_path = sys.argv[1]
    dest_path = sys.argv[2]

    for subfolder in ("train", "val", "test"):
        s_path = os.path.join(source_path, subfolder)
        d_path = os.path.join(dest_path, subfolder)
        os.makedirs(d_path, exist_ok=True)
        _copy_and_remove_overlap(s_path, d_path)

if __name__ == '__main__':
    run()
