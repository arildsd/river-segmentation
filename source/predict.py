import gdal
import tensorflow as tf
import model_utils
import numpy as np
import glob
import os



def evaluate_dataset_main():
    """
    A runable function to get a confusion matrix for a model on a validation set.
    :return: Nothing, prints a confusion matrix
    """

    # Validation data
    data_folder_path = r"/media/kitkat/Seagate Expansion Drive/Master_project/machine_learning_dataset/val"

    model_file_path = r"/home/kitkat/Master_project/runs/2020-01-30_17:08:11.173698_vgg16_freeze_first_no_augment/model.hdf5"
    model = model_utils.load_model(model_file_path)

    evaluate_dataset(model, data_folder_path)


def evaluate_dataset(model, data_folder_path):
    val = model_utils.load_dataset(data_folder_path)
    val_X, val_y = model_utils.convert_training_images_to_numpy_arrays(val)
    val_X = model_utils.fake_colors(val_X)

    model_utils.evaluate_model(model, val_X, val_y)


def predict_on_image(model, image_path):
    """
    Use the model to give a prediction on a image.
    :param model: A keras model.
    :param image_path: The path to a image in geotiff format. Image should be 512x512 and in black and white.
    :return: The prediction image as a TrainingImage object.
    """

    # Load image
    training_image = model_utils.load_data(image_path, image_path)
    data_X = model_utils.convert_training_images_to_numpy_arrays([training_image])[0]
    data_X = model_utils.fake_colors(data_X)

    prediction = model.predict(data_X)
    prediction = np.argmax(prediction, axis=-1)
    prediction = np.squeeze(prediction)

    training_image.labels = prediction

    return training_image


def predict_on_images(model, image_folder):
    paths = glob.glob(os.path.join(image_folder, "*.tif"))
    predictions = []
    for path in paths:
        predictions.append(predict_on_image(model, path))

    return predictions


def predict_on_images_main():
    model_path = r"/home/kitkat/Master_project/runs/2020-02-03_16:26:13.134402_vgg16_freeze_all_with_augment/model.hdf5"
    image_folder = r"/media/kitkat/Seagate Expansion Drive/Master_project/machine_learning_dataset_2/val/images"
    output_folder = r"/home/kitkat/Master_project/runs/2020-02-03_16:26:13.134402_vgg16_freeze_all_with_augment/predictions"

    model = model_utils.load_model(model_path)
    predictions = predict_on_images(model, image_folder)
    os.makedirs(output_folder, exist_ok=True)
    for pred in predictions:
        pred.write_labels_to_raster(os.path.join(output_folder, pred.name))


def predict_on_image_main():
    model_path = r"/home/kitkat/Master_project/runs/2020-01-31_12:29:39.620844_vgg16_freeze_first_with_augment/model.hdf5"
    image_path = r"/media/kitkat/Seagate Expansion Drive/Master_project/machine_learning_dataset/val/images/33-2-462-208-23_n_2048_e_512.tif"
    output_path = r"/home/kitkat/Master_project/runs/2020-01-31_12:29:39.620844_vgg16_freeze_first_with_augment/predictions/33-2-462-208-23_n_2048_e_512.tif"

    model = model_utils.load_model(model_path)
    image = predict_on_image(model, image_path)

    image.write_labels_to_raster(output_path)


def predict_and_evaluate(model_path, data_folder, output_folder):
    model = model_utils.load_model(model_path)

    predictions = predict_on_images(model, os.path.join(data_folder, "images"))
    os.makedirs(output_folder, exist_ok=True)
    for pred in predictions:
        pred.write_labels_to_raster(os.path.join(output_folder, pred.name))

    evaluate_dataset(model, data_folder)


def predict_and_evaluate_main():
    model_path = r"/home/kitkat/Master_project/runs/2020-02-02_23:47:09.414922_vgg16_freeze_all_no_augment/model.hdf5"
    data_folder = r"/media/kitkat/Seagate Expansion Drive/Master_project/machine_learning_dataset_2/val"
    output_folder = r"/home/kitkat/Master_project/runs/2020-02-02_23:47:09.414922_vgg16_freeze_all_no_augment/predictions"

    predict_and_evaluate(model_path, data_folder, output_folder)

if __name__ == '__main__':
    tf.keras.backend.clear_session()

    predict_and_evaluate_main()







