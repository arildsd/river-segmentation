from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import sys
import gdal

# GLOBAL CONSTANTS
UNKNOWN_CLASS_ID = 5

class TrainingImage:
    """
    A class for containing data (and metadata) for a training image.
    """

    def __init__(self, data, labels,  geo_transform, projection=None):
        if projection is None:
            projection = gdal.osr.SpatialReference()
            projection.ImportFromEPSG(25833)
        self.projection = projection
        self.geo_transform = geo_transform
        if data.shape != labels.shape:
            raise Exception(f"The shape of the data ({data.shape}) and labels ({labels.shape}) did not match")
        self.data = data
        self.labels = labels
        self.shape = data.shape

    def _write_array_to_raster(self, output_filepath, array):
        """
        Writes the given array to a raster image
        :param output_filepath: The output file
        :return: Nothing
        """

        driver = gdal.GetDriverByName("GTiff")
        raster = driver.Create(output_filepath, self.shape[0], self.shape[1],
                      1, gdal.GDT_Int16)
        raster.SetGeoTransform(self.geo_transform)
        raster.SetProjection(self.projection)
        raster.GetRasterBand(1).WriteArray(array)
        raster = None

    def write_data_to_raster(self, output_filepath):
        self._write_array_to_raster(output_filepath, self.data)

    def write_labels_to_raster(self, output_filepath):
        self._write_array_to_raster(output_filepath, self.labels)


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
    shape_0_indices = list(range(0, image_matrix.shape[0], image_size))
    shape_0_indices[-1] = image_matrix.shape[0]-image_size
    shape_1_indices = list(range(0, image_matrix.shape[1], image_size))
    shape_1_indices[-1] = image_matrix.shape[1] - image_size
    # Split the images
    for shape_0 in shape_0_indices:
        for shape_1 in shape_1_indices:
            labels = label_matrix[shape_0:shape_0 + image_size, shape_1:shape_1 + image_size]
            # Check if the entire image is of the unknown class, if so skip it
            is_unknown_matrix = labels == UNKNOWN_CLASS_ID
            if np.sum(is_unknown_matrix) == 0:
                continue
            data = image_matrix[shape_0:shape_0+image_size, shape_1:shape_1+image_size]
            new_geo_transform = list(geo_transform)
            new_geo_transform[0] += shape_1*geo_transform[1]  # East
            new_geo_transform[3] += shape_0*geo_transform[5]  # North
            training_data.append(TrainingImage(data, labels, new_geo_transform, projection=projection))
    return training_data


def conv_block(x, kernel_size=5, number_of_convolutions=3, filters=32, activation="relu"):
    for i in range(number_of_convolutions):
        x = keras.layers.Conv2D(filters, kernel_size,
                                activation=activation,
                                padding="same")(x)
    return x

def deconv_block(x, kernel_size=5, number_of_convolutions=3, filters=32, activation="relu"):
    for i in range(number_of_convolutions-1):
        x = keras.layers.Conv2DTranspose(filters, kernel_size,
                                         activation=activation, padding="same")(x)

    x = keras.layers.Conv2DTranspose(filters, kernel_size,
                                     activation=activation, padding="same",
                                     strides=(2, 2))(x)
    return x


def unet(input_shape, depth=3, kernel_size=5, number_of_convolutions=3, filters=32, activation="relu", n_classes=6):
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    skip_connections = []
    # Downsample
    for i, d in enumerate(range(depth)):
        x = conv_block(x, kernel_size=kernel_size,
                      number_of_convolutions=number_of_convolutions,
                      filters=filters,
                      activation=activation)

        skip_connections.append(x)
        x = keras.layers.MaxPooling2D()(x)

    # Upsample
    for d in reversed(range(depth)):
        x = deconv_block(x, kernel_size=kernel_size,
                      number_of_convolutions=number_of_convolutions,
                      filters=filters,
                      activation=activation)
        x = tf.concat([x, skip_connections[d]], -1)
    x = conv_block(x, kernel_size=kernel_size,
                      number_of_convolutions=number_of_convolutions,
                      filters=filters,
                      activation=activation)
    x = keras.layers.Conv2D(n_classes, (1, 1), activation="softmax", padding="same")(x)
    return inputs, x


if __name__ == '__main__':
    images = load_datapoint(r"D:\ortofoto\gaula_1963\33-2-462-210-11.tif",
                            r"D:\labels\rasters\gaula_1963\label33-2-462-210-11.tif")
    # Make dataset
    train_set_X = None
    train_set_y = None
    for i, image in enumerate(images):
        if i == 0:
            train_set_X = image.data
            train_set_X = np.expand_dims(train_set_X, 0)
            train_set_y = image.labels
            train_set_y = np.expand_dims(train_set_y, 0)
        else:
            train_set_X = np.concatenate([train_set_X, np.expand_dims(image.data, 0)], 0)
            train_set_y = np.concatenate([train_set_y, np.expand_dims(image.labels, 0)], 0)
    # Add channel axis
    train_set_X = np.expand_dims(train_set_X, -1)
    train_set_y = np.expand_dims(train_set_y, -1)


    inputs, outputs = unet(train_set_X.shape[1:])
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile("SGD", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.fit(train_set_X, train_set_y, epochs=10, batch_size=1)
    pass


