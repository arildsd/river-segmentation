from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import sys
import gdal
import data_processing


def load_data(image_path, label_path):
    # Load image
    image_ds = gdal.Open(image_path)
    geo_transform = image_ds.GetGeoTransform()
    projection = image_ds.GetProjection()
    image_matrix = image_ds.GetRasterBand(1).ReadAsArray()
    image_ds = None

    # Load label
    label_ds = gdal.Open(label_path)
    if label_ds.GetGeoTransform() != geo_transform:
        raise Exception(f"The geo transforms of image {image_path} and label {label_path} did not match")
    label_matrix = label_ds.GetRasterBand(1).ReadAsArray()
    label_ds = None

    training_image = data_processing.TrainingImage(image_matrix, label_matrix, geo_transform,
                                                   name=os.path.split(image_path)[-1], projection=projection)
    return training_image

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
    """

    :param input_shape:
    :param depth:
    :param kernel_size:
    :param number_of_convolutions:
    :param filters:
    :param activation:
    :param n_classes:
    :return:
    """
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    skip_connections = []
    # Downsample
    for d in range(depth):
        x = conv_block(x, kernel_size=kernel_size,
                      number_of_convolutions=number_of_convolutions,
                      filters=filters*(2**d),
                      activation=activation)

        skip_connections.append(x)
        x = keras.layers.MaxPooling2D()(x)

    # Upsample
    for d in reversed(range(depth)):
        x = deconv_block(x, kernel_size=kernel_size,
                      number_of_convolutions=number_of_convolutions,
                      filters=filters*(2**d),
                      activation=activation)
        x = tf.concat([x, skip_connections[d]], -1)
    x = conv_block(x, kernel_size=kernel_size,
                      number_of_convolutions=number_of_convolutions,
                      filters=filters,
                      activation=activation)
    x = keras.layers.Conv2D(n_classes, (1, 1), activation="softmax", padding="same")(x)
    return inputs, x


def sparse_Mean_IOU(y_true, y_pred):
    nb_classes = keras.backend.int_shape(y_pred)[-1]
    iou = []
    pred_pixels = keras.backend.argmax(y_pred, axis=-1)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = keras.backend.equal(y_true[:,:,0], i)
        pred_labels = keras.backend.equal(pred_pixels, i)
        inter = tf.dtypes.cast(true_labels & pred_labels, tf.int32)
        union = tf.dtypes.cast(true_labels | pred_labels, tf.int32)
        legal_batches = keras.backend.sum(tf.dtypes.cast(true_labels, tf.int32), axis=1)>0
        ious = keras.backend.sum(inter, axis=1)/keras.backend.sum(union, axis=1)
        iou.append(keras.backend.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.math.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return keras.backend.mean(iou)


def main(depth=3, kernel_size=5, number_of_convolutions=3, filters=32, activation="relu", momentum=0.99,
         learning_rate=0.01, drop_rate=0.5, n_classes=6):
    images = []
    # Load pointer files
    with open(r"D:\pointers\04\train.txt") as f:
        for line in f:
            line = line.replace("\n", "")
            image_path, label_path = line.split(";")
            images.append(load_data(image_path, label_path))
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
    # Normalize images to the range [0, 1]
    train_set_X = train_set_X / (2**8 - 1)  # 2**8 because of 8 bit encoding in original
    inputs, outputs = unet(train_set_X.shape[1:], depth=depth, kernel_size=kernel_size,
                           number_of_convolutions=number_of_convolutions, filters=filters, activation=activation,
                           n_classes=n_classes)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer, loss="sparse_categorical_crossentropy", metrics=[sparse_Mean_IOU])
    csv_logger = keras.callbacks.CSVLogger("training.log")
    history = model.fit(train_set_X, train_set_y, epochs=50, batch_size=8, callbacks=[csv_logger])



if __name__ == '__main__':
    main()
