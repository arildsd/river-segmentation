from tensorflow import keras
import numpy as np
import os
import sys
import gdal


class TrainingImage:
    def __init__(self, data, labels,  geo_transform, projection=None):
        if projection is None:
            projection = gdal.osr.SpatialReference()
            projection.ImportFromEPSG(25833)
        self.projection = projection
        self.geo_transform = geo_transform
        self.data = data
        self.labels = labels


def load_datapoint(image_filepath, label_filepath, image_size=512):
    # Load image
    image_ds = gdal.Open(image_filepath)
    geo_transform = image_ds.GetGeoTransform()
    projection = image_ds.GetProjection()
    image_matrix = image_ds.GetRasterBand(1).ReadAsArray()
    image_ds = None

    # Load label
    label_ds = gdal.Open(label_filepath)
    if label_ds.GetGeoTransform() != geo_transform:
        raise Exception(f"The geo transforms of image {image_filepath} and label {label_filepath} did not match")
    label_matrix = label_ds.GetRasterBand(1).ReadAsArray()
    label_ds = None

    training_data = []
    # Make properly sized training data
    # Make sure that the whole image is covered, even if the last one has to overlap
    shape_0_indices = list(range(0, image_size, image_matrix.shape[0]))
    if image_matrix.shape[0]-image_size not in shape_0_indices:
        shape_0_indices.append(image_matrix.shape[0]-image_size)
    shape_1_indices = list(range(0, image_size, image_matrix.shape[1]))
    if image_matrix.shape[1] - image_size not in shape_1_indices:
        shape_1_indices.append(image_matrix.shape[1] - image_size)
    # Split the images
    for shape_0 in shape_0_indices:
        for shape_1 in shape_1_indices:
            data = image_matrix[shape_0:shape_0+image_size, shape_1:shape_1+image_size]
            labels = label_matrix[shape_0:shape_0+image_size, shape_1:shape_1+image_size]
            training_data.append(TrainingImage(data, labels, geo_transform, projection=projection))
    return training_data


