import sklearn.metrics
import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import sys
import os
import random
import data_processing
import gdal

def load_data(image_path, label_path):
    # Load image
    image_ds = gdal.Open(image_path)
    geo_transform = image_ds.GetGeoTransform()
    projection = image_ds.GetProjection()
    image_matrix = image_ds.GetRasterBand(1).ReadAsArray()
    image_ds = None
    if np.isnan(np.min(image_matrix)):
        # The image contains a NaN value and will therefore be discarded
        return None

    # Load label
    label_ds = gdal.Open(label_path, gdal.GA_ReadOnly)
    label_matrix = label_ds.GetRasterBand(1).ReadAsArray()
    label_ds = None
    if np.isnan(np.min(label_matrix)):
        # The labels contains a NaN value and will therefore be discarded
        return None

    training_image = data_processing.TrainingImage(image_matrix, label_matrix, geo_transform,
                                                   name=os.path.split(image_path)[-1], projection=projection)
    return training_image


def load_dataset(data_folder_path):
    """
    A function to load an entire dataset given the data folder.
    :param data_folder_path: Path to the folder with a subfolders called images and labels. These subfolder should
    contain images in .tif format. A image in images should also have a corresponding image in labels with the same
    name.
    :return: The dataset as a list of TrainingImage objects.
    """

    file_paths = glob.glob(os.path.join(data_folder_path, "images", "*.tif"))
    file_path_endings = [os.path.split(path)[-1] for path in file_paths]
    data = []

    for ending in file_path_endings:
        image_path = os.path.join(data_folder_path, "images", ending)
        label_path = os.path.join(data_folder_path, "labels", ending)
        data.append(load_data(image_path, label_path))

    return data


def convert_training_images_to_numpy_arrays(training_images, one_hot_encode=False):
    """
    Converts the images from a list of TrainingImage objects to a numpy array for data and labels.
    :param training_images: A list of TrainingImage objects.
    :return: A tuple with (data_X, data_y) where data_X is a numpy array with shape (n, image_size, image_size, 1).
    (n is the number of images). The same applies to data_y but for the labels instead of images.
    """
    # Training set
    data_set_X = np.concatenate([np.expand_dims(image.data, 0) for image in training_images], 0)
    data_set_y = np.concatenate([np.expand_dims(image.labels, 0) for image in training_images], 0)
    # Add channel axis
    data_set_X = np.expand_dims(data_set_X, -1)
    data_set_y = np.expand_dims(data_set_y, -1)
    # Normalize images to the range [0, 1]
    data_set_X = data_set_X / (2 ** 8 - 1)  # 2**8 because of 8 bit encoding in original

    if one_hot_encode:
        data_set_y = tf.keras.utils.to_categorical(data_set_y, num_classes=6)

    return data_set_X, data_set_y


def fake_colors(data):
    """
    Adds copies of the first channel to two new channels to simulate a color image.
    :param data: A numpy array with shape (n, image_size, image_size, 1)
    :return: A numpy array with shape (n, image_size, image_size, 3)
    """

    new_data = np.concatenate([data, data, data], -1)
    return new_data




def image_augmentation(data):
    """
    Takes the original image matrix and add rotated images and mirrored images (with rotations).
    This adds 11 additional images for each original image.
    :param data:
    :return: An numpy array with the augmented images concatenated to the data array
    """
    rot_90 = np.rot90(data, axes=(1, 2))
    rot_180 = np.rot90(data, k=2, axes=(1, 2))
    rot_270 = np.rot90(data, k=3, axes=(1, 2))
    mirror = np.flip(data, axis=1)
    mirror_rot_90 = np.rot90(mirror, axes=(1, 2))
    mirror_rot_180 = np.rot90(mirror, k=2, axes=(1, 2))
    mirror_rot_270 = np.rot90(mirror, k=3, axes=(1, 2))
    mirror2 = np.flip(data, axis=2)
    mirror2_rot_90 = np.rot90(mirror2, axes=(1, 2))
    mirror2_rot_180 = np.rot90(mirror2, k=2, axes=(1, 2))
    mirror2_rot_270 = np.rot90(mirror2, k=3, axes=(1, 2))
    augments = [data, rot_90, rot_180, rot_270, mirror, mirror_rot_90, mirror_rot_180,
                mirror_rot_270, mirror2, mirror2_rot_90, mirror2_rot_180, mirror2_rot_270]
    augmented_image_matrix = np.concatenate(augments, axis=0)

    return augmented_image_matrix


def miou(y_true, y_pred, num_classes=6):
    """
    The intersection over union metric. Implemented with numpy.
    :param y_true: A flat numpy array with the true classes.
    :param y_pred: A flat numpy array with the predicted classes.
    :param num_classes: The number of classes.
    :return: The mean intersection over union.
    """
    ious = []
    for i in range(num_classes):
        y_true_class = y_true == i
        y_pred_class = y_pred == i
        intersection = np.sum(np.logical_and(y_true_class, y_pred_class))
        union = np.sum(np.logical_or(y_true_class, y_pred_class))
        ious.append(intersection/union)
    return np.sum(ious)/num_classes


def evaluate_model(model, data, labels):
    pred = model.predict(data, batch_size=1)
    pred = np.argmax(pred, axis=-1)

    f_labels = labels.flatten()
    f_pred = pred.flatten()

    conf_mat = sklearn.metrics.confusion_matrix(f_labels, f_pred)
    mean_intersection_over_union = miou(f_labels, f_pred)
    print(conf_mat)
    print(f"miou: {mean_intersection_over_union}")
    return conf_mat, mean_intersection_over_union

def load_model(model_file_path):
    """
    Loads the model at the file path. The model must include both architecture and weights
    :param model_file_path: The path to the model. (.hdf5 file)
    :return: The loaded model.
    """

    # Load the model
    model = tf.keras.models.load_model(model_file_path)
    return model


