import gdal
import numpy as np
import sys
import os
import glob

def mean(image_path):
    image_ds = gdal.Open(image_path)
    matrix = image_ds.GetRasterBand(1).ReadAsArray()
    image_ds = None

    array = matrix.flatten()
    mask = array > 0  # Remove invalid parts of the image
    array = array[mask]

    return np.mean(array), len(array)

def weighted_average(image_folder):
    tuples = []
    for image_path in glob.glob(os.path.join(image_folder, "*.tif")):
        tuples.append(mean(image_path))
    total = np.sum([t[1] for t in tuples])
    average = np.sum([t[0]*t[1] for t in tuples])/total
    return average

def tiny_image_mean(image_folders):
    image_paths = []
    for image_folder in image_folders:
        image_paths += glob.glob(os.path.join(image_folder, "*.tif"))
    means = []
    for image_path in image_paths:
        image_ds = gdal.Open(image_path)
        matrix = image_ds.GetRasterBand(1).ReadAsArray()
        means.append(np.mean(matrix.flatten()))

    return np.mean(means)


if __name__ == '__main__':
    image_folder = r"/media/kitkat/Seagate Expansion Drive/Master_project/machine_learning_dataset_6/test/images"
    print(tiny_image_mean([image_folder]))

