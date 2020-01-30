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
    label_ds = gdal.Open(label_path)
    if label_ds.GetGeoTransform() != geo_transform:
        raise Exception(f"The geo transforms of image {image_path} and label {label_path} did not match")
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
    data_set_X = None
    data_set_y = None
    for i, image in enumerate(training_images):
        if i == 0:
            data_set_X = image.data
            data_set_X = np.expand_dims(data_set_X, 0)
            data_set_y = image.labels
            data_set_y = np.expand_dims(data_set_y, 0)
        else:
            data_set_X = np.concatenate([data_set_X, np.expand_dims(image.data, 0)], 0)
            data_set_y = np.concatenate([data_set_y, np.expand_dims(image.labels, 0)], 0)
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


def sparse_Mean_IOU(y_true, y_pred):
    nb_classes = tf.keras.backend.int_shape(y_pred)[-1]
    iou = []
    pred_pixels = tf.keras.backend.argmax(y_pred, axis=-1)
    for i in range(0, nb_classes):
        true_labels = tf.keras.backend.equal(y_true[:,:,0], i)
        pred_labels = tf.keras.backend.equal(pred_pixels, i)
        inter = tf.dtypes.cast(true_labels & pred_labels, tf.int32)
        union = tf.dtypes.cast(true_labels | pred_labels, tf.int32)
        legal_batches = tf.keras.backend.sum(tf.dtypes.cast(true_labels, tf.int32), axis=1)>0
        ious = tf.keras.backend.sum(inter, axis=1)/tf.keras.backend.sum(union, axis=1)
        iou.append(tf.keras.backend.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.math.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return tf.keras.backend.sum(iou)/6


def evaluate_model(model, data, labels):
    pred = model.predict(data, batch_size=1)
    pred = np.argmax(pred, axis=-1)

    labels = np.argmax(labels, axis=-1)  # Convert to categorical

    conf_mat = sklearn.metrics.confusion_matrix(labels.flatten(), pred.flatten())
    print(conf_mat)
    return conf_mat

def load_model(model_file_path):
    """
    Loads the model at the file path. The model must include both architecture and weights
    :param model_file_path: The path to the model. (.hdf5 file)
    :return: The loaded model.
    """
    # Add the custom metric
    dependencies = {"sparse_Mean_IOU": sparse_Mean_IOU}

    # Load the model
    model = tf.keras.models.load_model(model_file_path, custom_objects=dependencies)
    return model


