import tensorflow as tf
import model_utils
from model_utils import sparse_Mean_IOU

if __name__ == '__main__':
    tf.keras.backend.clear_session()

    dependencies = {"sparse_Mean_IOU": sparse_Mean_IOU}

    model_file_path = r"/home/kitkat/PycharmProjects/river-segmentation/runs/2020-01-28 16:23:20.603404_vgg16_freeze_all_no_augment/model.hdf5"
    model = tf.keras.models.load_model(model_file_path, custom_objects=dependencies)


