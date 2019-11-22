from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import sys
import sklearn.metrics
import pandas as pd
import random
import glob


def create_pointer_files(data_path, output_folder, train_size=0.6, valid_size=0.2, test_size=0.2, shuffle=True,
                         sample_rate=1.0, file_ending="tif"):
    """
    DUPLICATED from data_processing to have access to this function without GDAL.
    Make txt files that point to images. Splitts into training, validation and test sets.
    :param output_folder:
    :param random_seed:
    :param data_path:
    :param train_size:
    :param valid_size:
    :param test_size:
    :param shuffle:
    :return:
    """
    total = train_size + test_size + valid_size
    if total != 1:
        raise Exception(f"The sizes don't sum to one, they sum to {total}")
    image_paths = glob.glob(os.path.join(data_path, "images", "*." + file_ending))
    if sample_rate < 1:
        image_paths = image_paths[:int(len(image_paths)*sample_rate)]

    if shuffle:
        random.shuffle(image_paths)
    label_paths = [path.replace("images", "labels").replace("tiny_labels", "tiny_images") for path in image_paths]
    os.makedirs(output_folder, exist_ok=True)
    # Make training file
    with open(os.path.join(output_folder, "train.txt"), "w+") as f:
        pairs = [image_paths[i] + ";" + label_paths[i] for i in range(int(train_size*len(image_paths)))]
        f.write("\n".join(pairs))
    # Make validation file
    with open(os.path.join(output_folder, "valid.txt"), "w+") as f:
        end_index = int(train_size * len(image_paths)) + int(valid_size * len(image_paths))
        pairs = [image_paths[i] + ";" + label_paths[i] for i in range(int(train_size * len(image_paths)), end_index)]
        f.write("\n".join(pairs))
    # Make test file
    with open(os.path.join(output_folder, "test.txt"), "w+") as f:
        start_index = int(train_size * len(image_paths)) + int(valid_size * len(image_paths))
        pairs = [image_paths[i] + ";" + label_paths[i] for i in range(start_index, len(image_paths))]
        f.write("\n".join(pairs))

def load_data(image_path, label_path):
    # Load image
    image_matrix = pd.read_csv(image_path, header=None)
    image_matrix = image_matrix.to_numpy(dtype=int)
    # Load labels
    label_matrix = pd.read_csv(label_path, header=False)
    label_matrix = label_matrix.to_numpy(dtype=int)

    return image_matrix, label_matrix


def load_dataset(pointer_file_path):
    image_tuples = []
    # Load pointer files
    with open(pointer_file_path) as f:
        for line in f:
            line = line.replace("\n", "")
            image_path, label_path = line.split(";")
            image_tuples.append(load_data(image_path, label_path))
    # Make dataset
    # Training set
    data_set_X = None
    data_set_y = None
    for i, image_tuple in enumerate(image_tuples):
        if i == 0:
            data_set_X = image_tuple[0]
            data_set_X = np.expand_dims(data_set_X, 0)
            data_set_y = image_tuple[1]
            data_set_y = np.expand_dims(data_set_y, 0)
        else:
            data_set_X = np.concatenate([data_set_X, np.expand_dims(image_tuple[0], 0)], 0)
            data_set_y = np.concatenate([data_set_y, np.expand_dims(image_tuple[1], 0)], 0)
    # Add channel axis
    data_set_X = np.expand_dims(data_set_X, -1)
    data_set_y = np.expand_dims(data_set_y, -1)
    # Normalize images to the range [0, 1]
    data_set_X = data_set_X / (2 ** 8 - 1)  # 2**8 because of 8 bit encoding in original

    return data_set_X, data_set_y


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
        x = keras.layers.Dropout(drop_rate)(x)
        x = keras.layers.Conv2D(filters, kernel_size,
                                activation=activation,
                                padding="same")(x)
    return x


def deconv_block(x, kernel_size=5, number_of_convolutions=3, filters=32, activation="relu", drop_rate=0.5):
    for i in range(number_of_convolutions-1):
        x = keras.layers.Dropout(drop_rate)(x)
        x = keras.layers.Conv2DTranspose(filters, kernel_size,
                                         activation=activation, padding="same")(x)

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
        do_test=False, test_set_X=None, test_set_y=None, patience=5, batch_size=8, logfile="training.log"):

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
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
    if do_validate:
        history = model.fit(train_set_X, train_set_y, validation_data=(valid_set_X, valid_set_y), epochs=50,
                            batch_size=batch_size, callbacks=[csv_logger, early_stopping])
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
                            batch_size=8, callbacks=[csv_logger, early_stopping])


def main(depth=3, kernel_size=5, number_of_convolutions=3, filters=32, activation="relu", momentum=0.0,
         learning_rate=0.001, drop_rate=0.5, n_classes=6, do_validate=True, do_test=False, patience=5, batch_size=8,
         logfile="training.log", do_image_augment=True):
    POINTER_FILE_PATH = r"~/pointers/05"
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
    data_path = "~/tiny_images_csv/05"
    dest_path = "~/pointers/05"
    create_pointer_files(data_path, dest_path, file_ending="csv")

    #main()
