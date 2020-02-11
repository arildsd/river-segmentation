import os
import shutil
import glob

if __name__ == '__main__':
    source_folder = r"/media/kitkat/Seagate Expansion Drive/Master_project/tiny_images_4"
    dest_folder = r"/media/kitkat/Seagate Expansion Drive/Master_project/machine_learning_dataset_4"
    big_image_list_file = r"/home/kitkat/PycharmProjects/river-segmentation/validation_images_list.txt"
    partition = "val"
    with open(big_image_list_file, "r") as f:
        big_image_list = [path.replace("\n", "") for path in f]

    images = []
    for big_image in big_image_list:
        images += glob.glob(os.path.join(source_folder, "images", big_image + "*.tif"))
    images = [os.path.split(path)[-1] for path in images]

    os.makedirs(os.path.join(dest_folder, partition, "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_folder, partition, "labels"), exist_ok=True)
    for path in images:
        shutil.copyfile(os.path.join(source_folder, "labels", path),
                        os.path.join(dest_folder, partition, "labels", path))
        shutil.copyfile(os.path.join(source_folder, "images", path),
                        os.path.join(dest_folder, partition, "images", path))
