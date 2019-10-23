import gdal
import numpy as np
import os

def translate_image(image, north_offset, east_offset, driver=gdal.GetDriverByName("GTiff")):
    """
    Translates the image north_offset meters to the north and east_offset meters to the east.

    :param image_path:
    :param north_offset:
    :param east_offset:
    :return:
    """


    geo_transform = image.GetGeoTransform()

    # Adapt the new geo transform
    gtl = list(geo_transform)
    gtl[0] += east_offset  # Move east
    gtl[3] += north_offset  # Move north

    # Apply the new geo transform
    image.SetGeoTransform(tuple(gtl))

    return image


def translate_all_in_dir(dir_input_path, dir_output_path, north_offset, east_offset,
                         driver=gdal.GetDriverByName("GTiff")):
    for filename in os.listdir(dir_input_path):
        if filename.endswith(".tif"):
            # Open the image and copy it
            original_image = gdal.Open(os.path.join(dir_input_path, filename))
            new_filepath = os.path.join(dir_output_path, f"translated_n{north_offset}_e{east_offset}_" + filename)
            new_image = driver.CreateCopy(new_filepath, original_image, strict=0)
            original_image = None  # Close the file the gdal way
            new_image = translate_image(new_image, north_offset, east_offset)  # Translate the file
            new_image = None  # Close the file the gdal way
        else:
            # Not a geo tiff and it will be ignored
            continue



if __name__ == '__main__':
    INPUT_DIR = r"../data/GAULA-1-OK"
    OUTPUT_DIR = r"../data/GAULA_1_OK_translated"
    north_offset = 53   # meters
    east_offset = -263  # meters

    translate_all_in_dir(INPUT_DIR, OUTPUT_DIR, north_offset, east_offset)


