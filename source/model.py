from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import sys
import gdal
import data_processing
import sklearn.metrics


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


def load_dataset(pointer_file_path):
    images = []
    # Load pointer files
    with open(pointer_file_path) as f:
        for line in f:
            line = line.replace("\n", "")
            image_path, label_path = line.split(";")
            image = load_data(image_path, label_path)
            if image is not None:
                images.append(image)
    # Make dataset
    # Training set
    data_set_X = None
    data_set_y = None
    for i, image in enumerate(images):
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

    return data_set_X, data_set_y


def balance_dataset(data):
    pass


def image_augmentation(data):
    """
    Takes the original image matrix and add rotated images and mirrored images (with rotations).
    This adds 7 additional images for each original image.
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
    augments = [data, rot_90, rot_180, rot_270, mirror, mirror_rot_90, mirror_rot_180, mirror_rot_270]
    augmented_image_matrix = np.concatenate(augments, axis=0)

    return augmented_image_matrix




def conv_block(x, kernel_size=5, number_of_convolutions=3, filters=32, activation="relu", drop_rate=0.5):
    for i in range(number_of_convolutions):
        if drop_rate > 0:
            x = keras.layers.Dropout(drop_rate)(x)
        x = keras.layers.Conv2D(filters, kernel_size,
                                activation=activation,
                                padding="same")(x)
    return x


def deconv_block(x, kernel_size=5, number_of_convolutions=3, filters=32, activation="relu", drop_rate=0.5):
    for i in range(number_of_convolutions-1):
        if drop_rate > 0:
            x = keras.layers.Dropout(drop_rate)(x)
        x = keras.layers.Conv2DTranspose(filters, kernel_size,
                                         activation=activation, padding="same")(x)

    if drop_rate > 0:
        x = keras.layers.Dropout(drop_rate)(x)
    x = keras.layers.Conv2DTranspose(filters, kernel_size,
                                     activation=activation, padding="same",
                                     strides=(2, 2))(x)
    return x


def unet(input_shape, depth=3, kernel_size=5, number_of_convolutions=3, filters=32, activation="relu", n_classes=6,
         drop_rate=0.5):
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
                      activation=activation,
                       drop_rate=drop_rate)

        skip_connections.append(x)
        x = keras.layers.MaxPooling2D()(x)

    # Upsample
    for d in reversed(range(depth)):
        x = deconv_block(x, kernel_size=kernel_size,
                      number_of_convolutions=number_of_convolutions,
                      filters=filters*(2**d),
                      activation=activation,
                      drop_rate=drop_rate)
        x = tf.concat([x, skip_connections[d]], -1)
    x = conv_block(x, kernel_size=kernel_size,
                      number_of_convolutions=number_of_convolutions,
                      filters=filters,
                      activation=activation,
                      drop_rate=drop_rate)
    x = keras.layers.Conv2D(n_classes, (1, 1), activation="softmax", padding="same")(x)
    return inputs, x


def sparse_Mean_IOU(y_true, y_pred):
    nb_classes = keras.backend.int_shape(y_pred)[-1]
    iou = []
    pred_pixels = keras.backend.argmax(y_pred, axis=-1)
    for i in range(0, nb_classes):
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
    return keras.backend.sum(iou)/6

def run(train_set_X, train_set_y, depth=3, kernel_size=5, number_of_convolutions=3, filters=32, activation="relu", momentum=0.0,
         learning_rate=0.001, drop_rate=0.5, n_classes=6, do_validate=True, valid_set_X=None, valid_set_y=None,
        do_test=False, test_set_X=None, test_set_y=None, patience=5, batch_size=4, logfile="training.log"):
    tf.keras.backend.clear_session()
    # Check for consistent input
    if do_validate and (valid_set_y is None or valid_set_X is None):
        raise Exception("Do validate was true but no validation set was given")
    if do_test and (test_set_y is None or test_set_X is None):
        raise Exception("do_test was true but no test set was given")

    inputs, outputs = unet(train_set_X.shape[1:], depth=depth, kernel_size=kernel_size,
                           number_of_convolutions=number_of_convolutions, filters=filters, activation=activation,
                           n_classes=n_classes)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer, loss="sparse_categorical_crossentropy", metrics=[sparse_Mean_IOU])
    # Prepare callbacks
    csv_logger = keras.callbacks.CSVLogger(logfile)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_sparse_Mean_IOU', patience=patience, mode='max')
    checkpoint = keras.callbacks.ModelCheckpoint(logfile.replace(".log", ".hdf5"),
                                                           monitor='val_sparse_Mean_IOU',
                                                           verbose=0, save_best_only=True,
                                                           save_weights_only=False, mode='max', period=1)

    if do_validate:
        history = model.fit(train_set_X, train_set_y, validation_data=(valid_set_X, valid_set_y), epochs=50,
                            batch_size=batch_size, callbacks=[csv_logger, early_stopping, checkpoint])
        val_pred = model.predict(valid_set_X, batch_size=batch_size)
        val_pred = np.argmax(val_pred, axis=-1)
        conf_mat = sklearn.metrics.confusion_matrix(valid_set_y.flatten(), val_pred.flatten())
        with open(logfile, "a") as f:
            f.write("Confusion matrix (validation set)\n")
            f.write(str(conf_mat))
            param_string = f"\ndepth={depth}, kernel_size={kernel_size}, number_of_convolutions={number_of_convolutions},"
            param_string += f" filters={filters}, activation={activation}, momentum={momentum}, "
            param_string += f"learning_rate={learning_rate}, drop_rate={drop_rate}, patience={patience}"
            f.write(param_string)
        print("Confusion matrix")
        print(conf_mat)
    else:
        history = model.fit(train_set_X, train_set_y, epochs=50,
                            batch_size=batch_size, callbacks=[csv_logger])
    return history


def main(depth=3, kernel_size=5, number_of_convolutions=3, filters=32, activation="relu", momentum=0.0,
         learning_rate=0.001, drop_rate=0.5, n_classes=6, do_validate=True, do_test=False, patience=5, batch_size=8,
         logfile="training.log", do_image_augment=True):
    POINTER_FILE_PATH = r"D:\pointers\04"
    # Train set
    train_set_X, train_set_y = load_dataset(os.path.join(POINTER_FILE_PATH, "train.txt"))
    valid_set_X, valid_set_y = None, None
    test_set_X, test_set_y = None, None
    if do_image_augment:
        train_set_X = image_augmentation(train_set_X)
        train_set_y = image_augmentation(train_set_y)
    if do_validate:
        valid_set_X, valid_set_y = load_dataset(os.path.join(POINTER_FILE_PATH, "valid.txt"))
    if do_test:
        test_set_X, test_set_y = load_dataset(os.path.join(POINTER_FILE_PATH, "test.txt"))

    run(train_set_X, train_set_y, depth=depth, kernel_size=kernel_size, number_of_convolutions=number_of_convolutions,
        filters=filters, activation=activation, momentum=momentum, learning_rate=learning_rate, drop_rate=drop_rate,
        n_classes=n_classes, do_validate=do_validate, do_test=do_test, patience=patience, batch_size=batch_size,
        valid_set_X=valid_set_X, valid_set_y=valid_set_y, test_set_X=test_set_X, test_set_y=test_set_y, logfile=logfile)




if __name__ == '__main__':
    main()
