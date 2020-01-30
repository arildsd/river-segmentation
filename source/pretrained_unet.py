import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import sys
import os
import random
import datetime
import model_utils

def vgg16_unet(image_size=512, n_max_filters=512, freeze="all"):
    """
    A unet model that uses a pre-trained VGG16 CNN as the encoder part.
    :param image_size: The size of the input images
    :param n_max_filters: The number of filters at the bottom layer of the unet model.
    :param freeze: Specifies what layers to freeze during training. The frozen layers will not be trained.
                all: all of the VGG16 layers are frozen. first: all but the last conv block of VGG16 is frozen.
                none: no layers are frozen
    :return:
    """

    # Determine what layers to freeze
    freeze_until = None
    if freeze == "all":
        freeze_until = 19
    elif freeze == "first":
        freeze_until = 15
    else:
        freeze_until = 0

    # Define input. It has 3 color channels since vgg is trained on a color dataset
    input = tf.keras.Input(shape=(image_size, image_size, 3))

    # Load pre-trained model
    vgg16 = tf.keras.applications.vgg16.VGG16(weights="imagenet",
                                              include_top=False, input_tensor=input)
    for i, layer in enumerate(vgg16.layers):
        if i < freeze_until:
            layer.trainable = False


    skip_connections = []

    # Get first conv block
    #input = vgg16.layers[0]  # Input layer in vgg
    x = vgg16.layers[1](input)  # Conv layer
    x = vgg16.layers[2](x)  # Conv layer
    skip_connections.append(x)
    x = vgg16.layers[3](x)  # Pooling layer

    # Get 2nd conv block
    x = vgg16.layers[4](x)  # Conv layer
    x = vgg16.layers[5](x)  # Conv layer
    skip_connections.append(x)
    x = vgg16.layers[6](x)  # Pooling layer

    # Get 3rd conv block
    x = vgg16.layers[7](x)  # Conv layer
    x = vgg16.layers[8](x)  # Conv layer
    x = vgg16.layers[9](x)  # Conv layer
    skip_connections.append(x)
    x = vgg16.layers[10](x)  # Pooling layer

    # Get 4th conv block
    x = vgg16.layers[11](x)  # Conv layer
    x = vgg16.layers[12](x)  # Conv layer
    x = vgg16.layers[13](x)  # Conv layer
    skip_connections.append(x)
    x = vgg16.layers[14](x)  # Pooling layer

    # # Get 5th conv block
    x = vgg16.layers[15](x)  # Conv layer
    x = vgg16.layers[16](x)  # Conv layer
    x = vgg16.layers[17](x)  # Conv layer
    skip_connections.append(x)
    x = vgg16.layers[18](x)  # Pooling layer

    # Starting upscaling and decoding
    for i, skip_i in enumerate(reversed(range(len(skip_connections)))):
        x = tf.keras.layers.Conv2D(int(n_max_filters/(2**i)), kernel_size=(3, 3),
                                   padding="same", activation="relu")(x)  # Conv layer
        x = tf.keras.layers.Conv2D(int(n_max_filters/(2**i)), kernel_size=(3, 3),
                                   padding="same", activation="relu")(x)  # Conv layer
        x = tf.keras.layers.Conv2DTranspose(int(n_max_filters/(2**i)), kernel_size=(3, 3), strides=2,
                                            padding="same", activation="relu")(x)  # Upsample
        x = tf.concat([x, skip_connections[skip_i]], axis=-1)  # Add skip connection to the channels

    # Last conv layers
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(6, kernel_size=(3, 3), padding="same", activation="softmax")(x)

    model = tf.keras.Model(inputs=input, outputs=x)

    return model


def run():
    #tf.keras.backend.clear_session()
    run_name = "vgg16_freeze_all_no_augment"
    run_path = "/home/kitkat/PycharmProjects/river-segmentation/runs"
    date = str(datetime.datetime.now())
    run_path = os.path.join(run_path, f"{date}_{run_name}")
    os.makedirs(run_path, exist_ok=True)

    # Load data
    # Training data
    data_folder_path = r"/media/kitkat/Seagate Expansion Drive/Master_project/machine_learning_dataset/train"
    train = model_utils.load_dataset(data_folder_path)
    train_X, train_y = model_utils.convert_training_images_to_numpy_arrays(train)
    train_X = model_utils.fake_colors(train_X)

    # Validation data
    data_folder_path = r"/media/kitkat/Seagate Expansion Drive/Master_project/machine_learning_dataset/val"
    val = model_utils.load_dataset(data_folder_path)
    val_X, val_y = model_utils.convert_training_images_to_numpy_arrays(val)
    val_X = model_utils.fake_colors(val_X)

    # Load and compile model
    model = vgg16_unet(freeze="all")
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Define callbacks
    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(patience=10, monitor="val_miou"))

    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(run_path, "model.hdf5"),
                                                    monitor="val_miou", save_best_only=True, mode="max")
    callbacks.append(checkpoint)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_path, histogram_freq=1)
    callbacks.append(tensorboard_callback)

    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(run_path, "log.csv"))
    callbacks.append(csv_logger)

    # Train the model
    model.fit(train_X, train_y, batch_size=1, epochs=10, validation_data=(val_X, val_y), callbacks=callbacks)

    # Print and save confusion matrix
    print("Confusion matrix on the validation data")
    conf_mat = model_utils.evaluate_model(model, val_X, val_y)
    with open(os.path.join(run_path, "conf_mat.txt"), "w+") as f:
        f.write(str(conf_mat))

if __name__ == '__main__':
    run()





