import os
import sys
import glob
import model_utils
import data_processing


"""
Augment and save the dataset. Augments are rotation and flipping. Fake colors are added too. Saved as .tif files. 
"""


if __name__ == '__main__':

    source_path = sys.argv[1]
    dest_path = sys.argv[2]

    # Load dataset
    image_paths = glob.glob(os.path.join(source_path, "*.tif"))
    for image_path in image_paths:
        image = model_utils.load_data(image_path, image_path.replace("images", "labels"))
        # Do preprocessing and image augmentation
        train_x, train_y = model_utils.convert_training_images_to_numpy_arrays([image], normalize=False)
        train_y = model_utils.replace_class(train_y, class_id=5)
        train_x = model_utils.fake_colors(train_x)
        train_x = model_utils.image_augmentation(train_x)
        train_y = model_utils.image_augmentation(train_y)

        # Save the images
        for i in range(train_x.shape[0]):
            augmented_image_x = train_x[i, :, :, :]
            augmented_image_y = train_y[i, :, :, :]
            augmented_image = data_processing.TrainingImage(augmented_image_x, augmented_image_y,
                                                            geo_transform=image.geo_transform,
                                                            projection=image.projection)
            data_output_path = os.path.join(dest_path, "images",
                                            os.path.split(image_path)[-1].replace(".tif", "") + f"_{i}.tif")
            augmented_image.write_data_to_raster(data_output_path)

            label_output_path = os.path.join(dest_path, "labels",
                                             os.path.split(image_path)[-1].replace(".tif", "") + f"_{i}.tif")
            augmented_image.write_labels_to_raster(label_output_path)

    print("Done!")
