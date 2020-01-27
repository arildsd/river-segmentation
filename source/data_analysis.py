from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import gdal
import sys
import os
import glob
import data_processing

def get_class_distribution(image, n_classes=6):
    """
    Gets the class distributions from a label image

    :param image: numpy array of the image
    :return: List with the num of each class
    """
    result = []
    for i in range(n_classes):
        is_class = image == i
        num_of_class = np.sum(is_class)
        result.append(num_of_class)
    return np.array(result)


def analyse_labels(image_dir):
    n_classes = 6
    image_paths = glob.glob(os.path.join(image_dir, "*.tif"))
    total_distribution = np.zeros(n_classes)
    for path in image_paths:
        # Extract pixels from the image
        image_ds = gdal.Open(path)
        image_array = image_ds.GetRasterBand(1).ReadAsArray()
        image_ds = None

        image_distribution = get_class_distribution(image_array, n_classes=n_classes)
        total_distribution += image_distribution
    # Convert from raw numbers to ratios
    total_distribution /= np.sum(total_distribution)
    print(total_distribution)

def analyse_filtering(label_dir):
    """
    Calculate the images removed by the different filtering steps
    :param label_dir: The path to the folder containing 512x512 images (raster labels)
    :return:
    """
    label_paths = glob.glob(os.path.join(label_dir, "*.tif"))
    mono_counter = 0
    unknown_class_counter = 0
    mono_unknown_class = 0
    for path in label_paths:
        label_ds = gdal.Open(path)
        label_matrix = label_ds.GetRasterBand(1).ReadAsArray()
        label_ds = None
        is_mono = False
        if data_processing.is_mono_class(label_matrix):
            mono_counter += 1
            is_mono = True
        is_unknown = False
        if data_processing.is_above_unknown_threshold(label_matrix):
            unknown_class_counter += 1
            is_unknown = True
        if is_mono and is_unknown:
            mono_unknown_class += 1
    print(f"Mono class images filtered away: {mono_counter}")
    print(f"Images with more than 10% of the unknown class: {unknown_class_counter}")
    print(f"Images with 100% of the unknown class: {mono_unknown_class}")



if __name__ == '__main__':
    IMAGE_DIR = r"/media/kitkat/Seagate Expansion Drive/Master_project/tiny_images_unfiltered/labels"
    analyse_filtering(IMAGE_DIR)

