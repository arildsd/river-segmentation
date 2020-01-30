import gdal
import tensorflow as tf
import model_utils
from model_utils import sparse_Mean_IOU
import numpy as np



def evaluate_dataset_main():
    """
    A runable function to get a confusion matrix for a model on a validation set.
    :return: Nothing, prints a confusion matrix
    """

    # Validation data
    data_folder_path = r"/media/kitkat/Seagate Expansion Drive/Master_project/machine_learning_dataset/val"
    val = model_utils.load_dataset(data_folder_path)
    val_X, val_y = model_utils.convert_training_images_to_numpy_arrays(val)
    val_X = model_utils.fake_colors(val_X)


    model_file_path = r"/home/kitkat/PycharmProjects/river-segmentation/runs/2020-01-28 16:55:47.858991_vgg16_freeze_all_no_augment/model.hdf5"
    model = model_utils.load_model(model_file_path)

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





if __name__ == '__main__':
    tf.keras.backend.clear_session()

    model_file_path = r"/home/kitkat/PycharmProjects/river-segmentation/runs/2020-01-28 16:55:47.858991_vgg16_freeze_all_no_augment/model.hdf5"
    model = model_utils.load_model(model_file_path)

    image_file_path = r"/media/kitkat/Seagate Expansion Drive/Master_project/machine_learning_dataset/val/images/33-2-462-209-11_n_2560_e_2048.tif"

    predict_on_image(model, image_file_path)







