from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import gdal
import sys
import os
import glob

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

if __name__ == '__main__':
    IMAGE_DIR = r"D:\tiny_images\05\labels"
    analyse_labels(IMAGE_DIR)
