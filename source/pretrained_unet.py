import tensorflow as tf
import numpy as np
import os
import datetime
import model_utils
import time
import resource


def vgg16_unet(image_size=512, n_max_filters=512, freeze="all", context_mode=False, dropout=0.0, num_classes=5):
    """
    A unet model that uses a pre-trained VGG16 CNN as the encoder part.
    :param num_classes: The number of classes
    :param image_size: The size of the input images
    :param n_max_filters: The number of filters at the bottom layer of the unet model.
    :param freeze: Specifies what layers to freeze during training. The frozen layers will not be trained.
                all: all of the VGG16 layers are frozen. first: all but the last conv block of VGG16 is frozen.
                none: no layers are frozen. number: freeze all conv blocks upto and including the number.
    :return: A keras model
    """

    # Determine what layers to freeze
    freeze = str(freeze).lower()
    freeze_until = None
    if freeze == "all":
        freeze_until = 19
    elif freeze == "first":
        freeze_until = 15
    elif freeze == "1":
        freeze_until = 3
    elif freeze == "2":
        freeze_until = 6
    elif freeze == "3":
        freeze_until = 10
    elif freeze == "4":
        freeze_until = 14
    elif freeze == "5":
        freeze_until = 18
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
        if dropout > 0: x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv2D(int(n_max_filters/(2**i)), kernel_size=(3, 3),
                                   padding="same", activation="relu")(x)  # Conv layer
        if dropout > 0: x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv2D(int(n_max_filters/(2**i)), kernel_size=(3, 3),
                                   padding="same", activation="relu")(x)  # Conv layer
        if dropout > 0: x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv2DTranspose(int(n_max_filters/(2**i)), kernel_size=(3, 3), strides=2,
                                            padding="same", activation="relu")(x)  # Upsample
        x = tf.concat([x, skip_connections[skip_i]], axis=-1)  # Add skip connection to the channels

    # Last conv layers
    if dropout > 0: x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(x)
    if dropout > 0: x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(x)
    if context_mode:
        # Crop to only predict on the middle pixels
        x = tf.keras.layers.MaxPool2D()(x)
    if dropout > 0: x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(num_classes, kernel_size=(3, 3), padding="same", activation="softmax")(x)

    model = tf.keras.Model(inputs=input, outputs=x)

    return model


def run(train_data_folder_path, val_data_folder_path, model_name="vgg16", freeze="all", image_augmentation=True,
        context_mode=False, run_path="/home/kitkat/PycharmProjects/river-segmentation/runs", replace_unknown=True,
        dropout=0):
    """
    Trains a CNN Unet model and saves the best model to file. If using large datasets consider using the run_from_dir
    function instead to decrease RAM usage.

    :param train_data_folder_path: Path to the folder containing training images (.tif format)
    :param val_data_folder_path: Path to the folder containing validation images (.tif format)
    :param model_name: The name of the model. Supported models are: vgg16
    :param freeze: Determine how many blocks in the encoder that are frozen during training.
    Should be all, first, 1, 2, 3, 4, 5 or none
    :param image_augmentation: Determines if image augmentation are used on the training data.
    :param context_mode: Determines if image context are included on the training data. Recommended set to False
    :param run_path: Folder where the run information and model will be saved.
    :param replace_unknown: When True the unknown class in the training date will be replaced using closest neighbor.
    :param dropout: Drop rate, [0.0, 1)
    :return: Writes model to the run folder, nothing is returned.
    """
    tf.keras.backend.clear_session()
    start_time = time.time()

    # Make run name based on parameters and timestamp
    augment = "with" if image_augmentation else "no"
    run_name = f"{model_name}_freeze_{freeze}_{augment}_augment"
    date = str(datetime.datetime.now())
    run_path = os.path.join(run_path, f"{date}_{run_name}".replace(" ", "_"))
    os.makedirs(run_path, exist_ok=True)

    # Load data
    # Training data
    train = model_utils.load_dataset(train_data_folder_path)
    print(f"Loading the training data took {time.time() - start_time} seconds")
    train_X, train_y = model_utils.convert_training_images_to_numpy_arrays(train)
    print(f"Converting to a numpy array took {time.time() - start_time} seconds")
    del train
    if replace_unknown:
        train_y = model_utils.replace_class(train_y, class_id=5)
    train_X = model_utils.fake_colors(train_X)
    if image_augmentation:
        train_X = model_utils.image_augmentation(train_X)
        train_y = model_utils.image_augmentation(train_y)
    print(f"Converting image augmentation and color faking took {time.time() - start_time} seconds")

    # Validation data
    val = model_utils.load_dataset(val_data_folder_path)
    val_X, val_y = model_utils.convert_training_images_to_numpy_arrays(val)
    del val
    if replace_unknown:
        model_utils.replace_class(val_y, class_id=5)
    val_X = model_utils.fake_colors(val_X)

    # Load and compile model
    if model_name.lower() == "vgg16":
        model = vgg16_unet(freeze=freeze, context_mode=context_mode, num_classes=5 if replace_unknown else 6,
                           dropout=dropout)
    else:
        model = None
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Define callbacks
    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(patience=10, monitor="val_loss"))

    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(run_path, "model.hdf5"),
                                                    monitor="val_loss", save_best_only=True)
    callbacks.append(checkpoint)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_path, histogram_freq=1)
    callbacks.append(tensorboard_callback)

    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(run_path, "log.csv"))
    callbacks.append(csv_logger)

    # Train the model
    model.fit(train_X, train_y, batch_size=4, epochs=100, validation_data=(val_X, val_y), callbacks=callbacks)

    # Print and save confusion matrix
    print("Confusion matrix on the validation data")
    conf_mat = model_utils.evaluate_model(model, val_X, val_y)
    with open(os.path.join(run_path, "conf_mat.txt"), "w+") as f:
        f.write(str(conf_mat))

    try:
        print("The current process uses the following amount of RAM (in GB) at its peak")
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
        print(resource.getpagesize())
    except Exception:
        print("Failed to print memory usage. This function was intended to run on a linux system.")


def run_from_dir(train_data_folder_path, val_data_folder_path, model_name="vgg16", freeze="all",
                 run_path="/home/kitkat/PycharmProjects/river-segmentation/runs", batch_size=1, dropout=0):
    """
        Trains a CNN Unet model and saves the best model to file. Uses training images from disk instead of loading
        everything into RAM.

        :param train_data_folder_path: Path to the folder containing training images (.png format)
        :param val_data_folder_path: Path to the folder containing validation images (.png format)
        :param model_name: The name of the model. Supported models are: vgg16
        :param freeze: Determine how many blocks in the encoder that are frozen during training.
        Should be all, first, 1, 2, 3, 4, 5 or none
        :param run_path: Folder where the run information and model will be saved.
        :param dropout: Drop rate, [0.0, 1)
        :return: Writes model to the run folder, nothing is returned.
        """

    tf.keras.backend.clear_session()
    start_time = time.time()

    # Make run name based on parameters and timestamp
    run_name = f"{model_name}_freeze_{freeze}"
    date = str(datetime.datetime.now())
    run_path = os.path.join(run_path, f"{date}_{run_name}".replace(" ", "_"))
    os.makedirs(run_path, exist_ok=True)

    # Setup data generators
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=lambda x: x/(2**8 -1))
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    image_generator = image_datagen.flow_from_directory(os.path.join(train_data_folder_path, "images"),
                                                        class_mode=None, target_size=(512, 512), seed=1, batch_size=batch_size)
    mask_generator = mask_datagen.flow_from_directory(os.path.join(train_data_folder_path, "labels"),
                                                      class_mode=None, target_size=(512, 512), seed=1, batch_size=batch_size,
                                                      color_mode="grayscale")
    train_generator = (pair for pair in zip(image_generator, mask_generator))

    # Validation data
    val = model_utils.load_dataset(val_data_folder_path)
    val_X, val_y = model_utils.convert_training_images_to_numpy_arrays(val)
    val_X = model_utils.fake_colors(val_X)
    val_y = model_utils.replace_class(val_y, class_id=5)

    # Load and compile model
    if model_name.lower() == "vgg16":
        model = vgg16_unet(freeze=freeze, context_mode=False, num_classes=5, dropout=dropout)
    else:
        model = None
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Define callbacks
    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(patience=10, monitor="val_loss"))

    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(run_path, "model.hdf5"),
                                                    monitor="val_loss", save_best_only=True)
    callbacks.append(checkpoint)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=run_path, histogram_freq=1)
    callbacks.append(tensorboard_callback)

    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(run_path, "log.csv"))
    callbacks.append(csv_logger)

    # Train the model
    model.fit_generator(train_generator, epochs=100, validation_data=(val_X, val_y), steps_per_epoch=int(np.ceil(57648/batch_size)), callbacks=callbacks, verbose=2)

    # Print and save confusion matrix
    print("Confusion matrix on the validation data")
    conf_mat = model_utils.evaluate_model(model, val_X, val_y)
    with open(os.path.join(run_path, "conf_mat.txt"), "w+") as f:
        f.write(str(conf_mat))

    try:
        print("The current process uses the following amount of RAM (in GB) at its peak")
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20)
        print(resource.getpagesize())
    except Exception:
        print("Failed to print memory usage. This function was intended to run on a linux system.")


if __name__ == '__main__':
    model_name = "vgg16"
    freeze = "first"
    image_augmentation = True
    context_mode = False
    run_path = "/home/kitkat/PycharmProjects/river-segmentation/runs"

    train_data_folder_path = r"/media/kitkat/Seagate Expansion Drive/Master_project/machine_learning_dataset/train"
    val_data_folder_path = r"/media/kitkat/Seagate Expansion Drive/Master_project/machine_learning_dataset/val"

    run(train_data_folder_path, val_data_folder_path, model_name=model_name, freeze=freeze,
        image_augmentation=image_augmentation, context_mode=context_mode, replace_unknown=True)






