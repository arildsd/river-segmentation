import model_utils
import data_processing
import glob
import os
import sys
import numpy as np
import gdal

"""
This script should be used to make predictions on a set of big images (6000x8000 pixels). Image of other sizes should 
also work but this has not been tested. A folder of images must be provided as well as a trained model. 
The predictions will be written to a output folder. The output images will be in
the big image format as geo referenced tiff files. 
"""


def run(model_path, input_folder, output_folder, intensity_correction=0.0):
    model = model_utils.load_model(model_path)
    big_image_paths = glob.glob(os.path.join(input_folder, "*.tif"))
    for big_image_path in big_image_paths:
        images = data_processing.divide_image(big_image_path, big_image_path, image_size=512, do_crop=False,
                                              do_overlap=False)
        # Make predictions
        for image in images:
            data = model_utils.convert_training_images_to_numpy_arrays([image])[0]
            data += intensity_correction / (2**8 - 1)
            data = model_utils.fake_colors(data)

            prediction = model.predict(data)
            prediction = np.argmax(prediction, axis=-1)
            prediction = np.squeeze(prediction)
            image.labels = prediction

        big_image_ds = gdal.Open(big_image_path)
        geo_transform = big_image_ds.GetGeoTransform()
        projection = big_image_ds.GetProjection()
        big_image_shape = (big_image_ds.RasterYSize, big_image_ds.RasterXSize)
        big_image_ds = None  # Close the image the gdal way

        big_image_array = data_processing.reassemble_big_image(images, small_image_size=512,
                                                               big_image_shape=big_image_shape)
        big_image = data_processing.TrainingImage(big_image_array, big_image_array, geo_transform,
                                                  projection=projection, name=os.path.split(big_image_path)[-1])
        big_image.write_labels_to_raster(os.path.join(output_folder, big_image.name))


if __name__ == '__main__':
    # Get args
    model_path = sys.argv[1]
    input_folder = sys.argv[2]
    output_folder = sys.argv[3]
    if len(sys.argv) >= 5:
        intensity_correction = float(sys.argv[4])
    else:
        intensity_correction = 0.0
    # Predict and write to file
    run(model_path, input_folder, output_folder, intensity_correction=intensity_correction)