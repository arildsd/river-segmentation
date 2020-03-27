import numpy as np
import gdal
import json
import glob
import os
import sys
from osgeo import ogr
from osgeo import osr
import copy
import random
import pandas as pd
import shutil

"""
Look here for tips:
https://pcjericks.github.io/py-gdalogr-cookbook/geometry.html
"""

# GLOBAL CONSTANTS
UNKNOWN_CLASS_ID = 5

class TrainingImage:
    """
    A class for containing data (and metadata) for a training image.
    """

    def __init__(self, data, labels, geo_transform, name="", projection=None, label_geo_transform=None,
                 north_offset=None, east_offset=None):
        """
        A class to store the data and corresponding labels with methods for writing to file.

        :param data: Image in the form of an numpy array with shape (image_size, image_size)
        :param labels: Label in the form of an numpy array with shape (image_size, image_size).
        The array should be filled with the class id for that pixel.
        :param geo_transform: A list defining the geo transform. See gdal doc for more info
        :param name: The name of the image.
        :param projection: The Geo projection. See gdal doc for more info.
        :param label_geo_transform: The geo transform of the label.
        :param north_offset: The offset in pixels from the most north point when dividing the image.
        :param east_offset: The offset in pixels from the most east point when dividing the image.
        """

        if projection is None:
            projection = gdal.osr.SpatialReference()
            projection.ImportFromEPSG(25833)
        self.projection = projection
        self.geo_transform = geo_transform
        self.label_geo_transform = label_geo_transform
        self.data = data
        self.labels = labels
        self.name = name
        self.north_offset = north_offset
        self.east_offset = east_offset

    def _write_array_to_raster(self, output_filepath, array, geo_transform):
        """
        Writes the given array to a raster image
        :param output_filepath: The output file
        :return: Nothing
        """
        bands = array.shape[-1] if len(array.shape) > 2 else 1
        driver = gdal.GetDriverByName("GTiff")
        raster = driver.Create(output_filepath, array.shape[1], array.shape[0],
                      bands, gdal.GDT_Int16)
        raster.SetGeoTransform(geo_transform)
        raster.SetProjection(self.projection)
        # Write multiple bands
        if len(array.shape) > 2:
            for band in range(array.shape[-1]):
                raster.GetRasterBand(band + 1).WriteArray(array[:,:,band])
        # Write single band
        else:
            raster.GetRasterBand(1).WriteArray(array)
        raster = None

    def write_data_to_raster(self, output_filepath):
        self._write_array_to_raster(output_filepath, self.data, self.geo_transform)

    def write_labels_to_raster(self, output_filepath):
        if self.label_geo_transform is not None:
            geo_transform = self.label_geo_transform
        else:
            geo_transform = self.geo_transform
        self._write_array_to_raster(output_filepath, self.labels, geo_transform)


def create_pointer_files(data_path, output_folder, train_size=0.6, valid_size=0.2, test_size=0.2, shuffle=True,
                         sample_rate=1.0, file_ending="tif"):
    """
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


def is_above_unknown_threshold(label_image, unknown_threshold=0.1):
    # Check that the amount of the unknown class
    unknown_label_matrix = label_image == UNKNOWN_CLASS_ID
    if np.sum(unknown_label_matrix) > unknown_threshold * unknown_label_matrix.shape[0] * unknown_label_matrix.shape[1]:
        return True
    else:
        return False

def is_mono_class(label_image):
    for i in range(6):
        is_mono_class = True
        for e in np.nditer(label_image):
            if e != i:
                is_mono_class = False
                break
        # The image had only one class
        if is_mono_class:
            return True
    return False

def is_quality_image(label_image, unknown_threshold=0.05):
    """
    Check that the image don't contain to much of the unknown class, as this is an indication of missing (wrong) labels
    :param label_image: A numpy matrix representing the label image
    :return: bool, true if the image passes the check, 0 otherwise
    """
    # Discard images with more than a threshold of the unknown class
    if is_above_unknown_threshold(label_image, unknown_threshold):
        return False
    # Discard images with only one class
    if is_mono_class(label_image):
        return False

    return True


def divide_image(image_filepath, label_filepath, image_size=512, do_overlap=False, do_crop=False):
    # Load image
    image_ds = gdal.Open(image_filepath)
    geo_transform = image_ds.GetGeoTransform()
    projection = image_ds.GetProjection()
    image_matrix = image_ds.GetRasterBand(1).ReadAsArray()
    image_ds = None

    # Load label
    label_ds = gdal.Open(label_filepath)
    if label_ds.GetGeoTransform() != geo_transform:
        raise Exception(f"The geo transforms of image {image_filepath} and label {label_filepath} did not match")
    label_matrix = label_ds.GetRasterBand(1).ReadAsArray()
    label_ds = None

    training_data = []
    # Make properly sized training data
    # Make sure that the whole image is covered, even if the last one has to overlap
    if do_overlap:
        shape_0_indices = list(range(image_size // 4, image_matrix.shape[0], image_size // 4))[:-4]
        shape_1_indices = list(range(image_size // 4, image_matrix.shape[1], image_size // 4))[:-4]
    else:
        shape_0_indices = list(range(0, image_matrix.shape[0], image_size))
        shape_0_indices[-1] = image_matrix.shape[0] - image_size
        shape_1_indices = list(range(0, image_matrix.shape[1], image_size))
        shape_1_indices[-1] = image_matrix.shape[1] - image_size
    # Split the images
    for shape_0 in shape_0_indices:
        for shape_1 in shape_1_indices:
            if do_crop:
                # Extract labels for the center of the image
                labels = label_matrix[shape_0 + image_size // 4:shape_0 + image_size - image_size // 4,
                         shape_1 + image_size // 4:shape_1 + image_size - image_size // 4]
            else:
                labels = label_matrix[shape_0:shape_0 + image_size, shape_1:shape_1 + image_size]
            # Check if the image has to much unknown
            if not is_quality_image(labels):
                continue

            # Calculate the geo transform of the label
            label_geo_transform = list(geo_transform)
            if do_crop:
                label_geo_transform[0] += (shape_1 + image_size//4) * geo_transform[1]  # East
                label_geo_transform[3] += (shape_0 + image_size//4) * geo_transform[5]  # North
            else:
                label_geo_transform[0] += (shape_1) * geo_transform[1]  # East
                label_geo_transform[3] += (shape_0) * geo_transform[5]  # North

            data = image_matrix[shape_0:shape_0 + image_size, shape_1:shape_1 + image_size]
            new_data_geo_transform = list(geo_transform)
            new_data_geo_transform[0] += shape_1 * geo_transform[1]  # East
            new_data_geo_transform[3] += shape_0 * geo_transform[5]  # North

            name = os.path.split(image_filepath)[-1].replace(".tif", "") + f"_n_{shape_0}_e_{shape_1}"
            training_data.append(TrainingImage(data, labels, new_data_geo_transform, name=name, projection=projection,
                                               label_geo_transform=label_geo_transform, east_offset=shape_1,
                                               north_offset=shape_0))
    return training_data


def divide_and_save_images(image_filepaths, label_filepaths, output_folder=None, image_size=512,
                           do_overlap=False, do_crop=False):
    """
    This function takes big images and splits them into smaller images and saves them to disk
    :param image_filepaths: A list of filepaths to the images that will be loaded
    :param label_filepaths: A list of filepaths to the label (rasters) that will be loaded
    :param output_folder: The folder where the new rasters will be saved. If it is None, the files will not be saved
    :param image_size: The size of the new images, measured in pixels
    :return: list of TrainingImage objects
    """

    # Check that the size of the filepaths are the same are loaded
    if len(image_filepaths) != len(label_filepaths):
        raise Exception(f"The image filepaths and label filepaths must be in sync,"
                        f" but their lengths did not match. {len(image_filepaths)} != {len(label_filepaths)}")
    # Load the images
    # Write the images to disk
    if output_folder is not None:
        # Make output folders
        os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "labels"), exist_ok=True)
        for i in range(len(image_filepaths)):
            training_images = divide_image(image_filepaths[i], label_filepaths[i], image_size=image_size,
                                           do_overlap=do_overlap, do_crop=do_crop)
            for image in training_images:
                # Data
                data_path = os.path.join(output_folder, "images", image.name + ".tif")
                image.write_data_to_raster(data_path)
                # Labels
                label_path = os.path.join(output_folder, "labels", image.name + ".tif")
                image.write_labels_to_raster(label_path)


def find_intersecting_polys(geometry, polys):
    intersecting_polys = []
    for poly in polys:
        if poly.Intersects(geometry):
            intersecting_polys.append(poly)
    return intersecting_polys


def rasterize_polygons(polygons, image_path, class_name, shapefile_path, driver=gdal.GetDriverByName("MEM")):
    """
    Retuns a data set image with the bounding box dimensions and the polygon locations marked by 1.
    :param polygons: A list of polygons of the same class
    :param image_path The path to the image
    :return: Data set image
    """
    # Get meta data from the image
    image_ds = gdal.Open(image_path)
    geo_transform = image_ds.GetGeoTransform()
    projection = image_ds.GetProjection()
    n_pixels_north = image_ds.RasterYSize
    n_pixels_east = image_ds.RasterXSize
    image_ds = None
    # Create empty image
    output_path = image_path.replace(".tif", "")
    output_path += f"_temp_label_{class_name}.tif"
    label_raster = driver.Create(output_path, n_pixels_east, n_pixels_north,
                  1, gdal.GDT_Int16)
    label_raster.SetGeoTransform(geo_transform)
    label_raster.SetProjection(projection)

    # set up the shapefile driver
    shapefile_driver = ogr.GetDriverByName("Memory")
    # create the data source
    poly_ds = shapefile_driver.CreateDataSource(shapefile_path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(25833)
    layer = poly_ds.CreateLayer(os.path.split(shapefile_path.replace(".shp", ""))[-1],
                                srs, ogr.wkbPolygon)
    # Add the polygons to the new layer
    for poly in polygons:
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(poly)
        layer.CreateFeature(feature)
        feature.Destroy()

    # Burn the polygons into the new image
    gdal.RasterizeLayer(label_raster, (1,), layer, burn_values=(1,))
    poly_ds.Destroy()

    # Save and close the image
    return label_raster


def find_closest_pixel(i, j, arrays, threshold=10):
    """

    :param i: shape 0 axis index
    :param j: shape 1 axis index
    :param arrays: arrays with labels
    :param threshold: The (max) distance in pixels that are searched
    :return: The class of the closest pixel (majority if there is more than one)
    """
    # Find shape
    shape = None
    for array in arrays:
        if array is not None:
            shape = array.shape
            break
    # Find dimensions of the box
    distance_to_closest_edge = min(i, j, shape[0]-i-1, shape[1]-j-1)
    max_radius = min(threshold, distance_to_closest_edge)
    # Check if there are any classes present
    id_count = [0] * len(arrays)
    total = 0
    for identifier, array in enumerate(arrays):
        if array is not None:
            id_count[identifier] += np.sum(array[i-max_radius:i+max_radius, j-max_radius:j+max_radius])
    if sum(id_count) > 0:
        return np.argmax(id_count)
    else:
        # No class was in the search area, return the ID of the unknown class
        return 5

    # Unreachable code on purpose, it was to slow
    radius = 0
    while radius < max_radius:
        ids = []
        radius += 1
        for search_i in range(i-radius, i+radius+1):
            for search_j in range(j-radius, j+radius+1):
                for identifier, array in enumerate(arrays):
                    if array is None: continue
                    if array[search_i][search_j] > 0:
                        ids.append(identifier)
        id_count = [0]*len(arrays)
        # Find the majority id
        for id in ids:
            id_count[id] += 1
        return np.argmax(id_count)
    # Return the id of the "unknown" class since no other classes where found
    return 5


def reassemble_big_image(images, small_image_size=512):
    big_image = np.zeros((6000, 8000)) + 5
    for image in images:
        image_offset_shape_0 = image.north_offset
        image_offset_shape_1 = image.east_offset
        big_image[image_offset_shape_0:image_offset_shape_0+small_image_size,
                  image_offset_shape_1:image_offset_shape_1+small_image_size] = image.labels
    return big_image

def merge_labels_rasters(label_raster_dict):
    """

    (array = None is equivalent with a zero array but is kept as None to save computation time)
    :param label_raster_dict:
    :return:
    """
    arrays = []
    s_IDs = sorted(label_raster_dict.keys())
    for id in s_IDs:
        if label_raster_dict[id] is None:
            array = None
        else:
            array = label_raster_dict[id].GetRasterBand(1).ReadAsArray()
        arrays.append(array)
    # Check array shapes, they should all be the same
    shape = None
    for array in arrays:
        if array is not None:
            shape = array.shape
            break
    for array in arrays:
        if array is None: continue
        if array.shape != shape:
            raise Exception(f"The shapes does not match, {shape} != {array.shape}")
    label_matrix = np.zeros(shape, dtype=int)
    for i in range(shape[0]):
        for j in range(shape[1]):
            ids_at_pixel = []
            for id, array in enumerate(arrays):
                if array is None:
                    continue
                else:
                    pixel = array[i][j]
                    if pixel > 0:
                        ids_at_pixel.append(id)
            # Only one class at the location
            if len(ids_at_pixel) == 1:
                label_matrix[i][j] = ids_at_pixel[0]
            # No pixels matches, will fill in with closest value
            elif len(ids_at_pixel) == 0:
                label_matrix[i][j] = find_closest_pixel(i, j, arrays)
            # Multiple pixels at the same loc, take the majority
            elif len(ids_at_pixel) > 1:
                id_count = [0] * len(arrays)
                # Find the majority id
                for id in ids_at_pixel:
                    id_count[id] += 1
                label_matrix[i][j] = np.argmax(id_count)
            else:
                raise Exception("Something weird happened...")
    return label_matrix


def create_raster_labels(image_path, poly_dict, destination_path, driver=gdal.GetDriverByName("GTiff")):
    """
    Creates a raster with all the polygons as pixel values.
    :param image_path: Path to the image that defines the bounding box
    :param poly_dict: A dict with class_ID -> class_polygons
    :return:
    """
    # Skip if the image has a label image already
    if os.path.isfile(destination_path):
        print(f"{destination_path} already exists")
        return None

    # Create bounding box for the image
    image_ds = gdal.Open(image_path)
    bounding_box = create_bounding_box(image_ds)
    label_raster_dict = {}
    for current_class in poly_dict:
        polys = poly_dict[current_class]
        # Find intersecting polygons
        intersecting_polys = find_intersecting_polys(bounding_box, polys)
        if len(intersecting_polys) == 0:
            label_raster_dict[current_class] = None
        else:
            shapefile_path = r"D:\temp\post_processed_label_" + os.path.split(destination_path.replace(".tif", ".shp"))[-1]
            label_raster_dict[current_class] = rasterize_polygons(intersecting_polys, image_path,
                                                                  current_class, shapefile_path)
    # Check if there is any polygons in the image
    have_overlap = False
    for v in label_raster_dict.values():
        if v is not None:
            have_overlap = True
            break
    if not have_overlap:
        # There was no overlapping polygons so there is no point in making a raster for it
        return None
    # Merge the different class layer to one
    label_matrix = merge_labels_rasters(label_raster_dict)
    label_dataset = driver.Create(destination_path, image_ds.RasterXSize, image_ds.RasterYSize,
                                  1, gdal.GDT_Int16)
    label_dataset.SetGeoTransform(image_ds.GetGeoTransform())
    label_dataset.SetProjection(image_ds.GetProjection())
    label_dataset.GetRasterBand(1).WriteArray(label_matrix)
    # Save, the gdal way
    label_dataset = None
    print(f"Wrote label image {image_path} to {destination_path}")


def burn_labels_to_image(image_path, shapefile_path, class_id):
    """
    Burn labels from the shapefile on the image. Will only override the values where the shapefile intersects, the rest
    will remain unchanged.

    :param image_path: Path to the raster label image.
    :param shapefile_path: Path to the shapefile.
    :param int: The class id of the shapefile.
    :return: Write the image to file (overriding). Return nothing.
    """
    # Get meta data from the image
    image_ds = gdal.Open(image_path, gdal.GA_Update)
    geo_transform = image_ds.GetGeoTransform()
    projection = image_ds.GetProjection()
    n_pixels_north = image_ds.RasterYSize
    n_pixels_east = image_ds.RasterXSize

    # Get geometries from shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.Open(shapefile_path, 0)
    layer = ds.GetLayer()
    gdal.RasterizeLayer(image_ds, (1,), layer, burn_values=(class_id,))


def create_bounding_box(image_ds):
    # Create a bounding box geometry for the image
    geo_transform = image_ds.GetGeoTransform()
    projection = image_ds.GetProjection()
    n_pixels_north = image_ds.RasterYSize
    n_pixels_east = image_ds.RasterXSize
    top_left_coordinate = (geo_transform[0], geo_transform[3])
    bottom_right_coordinate = (geo_transform[0] + n_pixels_east * geo_transform[1],
                               geo_transform[3] + geo_transform[5] * n_pixels_north)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(top_left_coordinate[0], top_left_coordinate[1])  # Top left
    ring.AddPoint(top_left_coordinate[0], bottom_right_coordinate[1])  # Top right
    ring.AddPoint(bottom_right_coordinate[0], bottom_right_coordinate[1])  # Bottom right
    ring.AddPoint(bottom_right_coordinate[0], top_left_coordinate[1])  # Bottom left
    ring.AddPoint(top_left_coordinate[0], top_left_coordinate[1])  # Top left, closes the ring
    rectangle = ogr.Geometry(ogr.wkbPolygon)
    rectangle.AddGeometry(ring)
    ref = osr.SpatialReference()
    ref.ImportFromEPSG(25833)
    rectangle.AssignSpatialReference(ref)
    return rectangle


def name_to_id(name):
    name = name.lower()
    if name == "water":
        return 0
    elif name == "gravel":
        return 1
    elif name == "vegetation":
        return 2
    elif name == "farmland":
        return 3
    elif name == "human-constructions" or name == "human-construction":
        return 4
    elif name == "undefined":
        return 5
    else:
        print(f"WARNING: could not assign the name {name} to an id")
        return None


def load_polygons(folder_path):
    """
    Loads all the shapefiles in the folder into gdal geometries.
    :param folder_path: The path to the shapefile folder
    :return: A dict with ID -> label geometry
    """
    id_poly_dict = {}
    filepaths = glob.glob(os.path.join(folder_path, "*.shp"))
    print(f"Filepaths: {filepaths}")
    for path in filepaths:
        # Get the ID corresponding to the name
        last_part_of_path = os.path.split(path)[-1].replace(".shp", "")
        identifier = name_to_id(last_part_of_path.split("_")[-1])

        # Load the shapefile into a geometry
        driver = ogr.GetDriverByName("ESRI Shapefile")
        ds = driver.Open(path, 0)
        layer = ds.GetLayer()
        polys = []
        for feature in layer:
            geom = feature.GetGeometryRef().Clone()
            ref = osr.SpatialReference()
            ref.ImportFromEPSG(25833)
            geom.AssignSpatialReference(ref)
            polys.append(geom)
        id_poly_dict[identifier] = polys
    return id_poly_dict


def convert_to_CSV(pointer_file_path, dest_dir):
    """
    Saves the images in pointer file that are saved in .tif format as .csv to remove need for Gdal for data loading

    :param pointer_file_path:
    :param dest_dir:
    :return:
    """
    # Load pointer files
    with open(pointer_file_path, "r") as f:
        for line in f:
            line = line.replace("\n", "")
            image_path, label_path = line.split(";")
            # Load images
            image_ds = gdal.Open(image_path)
            image = image_ds.GetRasterBand(1).ReadAsArray()
            image_ds = None
            label_ds = gdal.Open(label_path)
            label = label_ds.GetRasterBand(1).ReadAsArray()
            label_ds = None
            # Write images as csv
            pd.DataFrame(image).to_csv(os.path.join(dest_dir, "images",
                                                    os.path.split(image_path)[-1].replace(".tif", ".csv")),
                                       index=False, header=False)
            pd.DataFrame(label).to_csv(os.path.join(dest_dir, "labels",
                                                    os.path.split(label_path)[-1].replace(".tif", ".csv")),
                                       index=False, header=False)


def convert_many_to_CSV(pointer_file_dir, dest_dir):
    pointer_files = glob.glob(os.path.join(pointer_file_dir, "*.txt"))
    for pointer_file_path in pointer_files:
        convert_to_CSV(pointer_file_path, dest_dir)
    print(f"Converted all images in {pointer_files} to csv \nand saved them in {dest_dir}")


def process_and_rasterize_raw_data():
    gdal.UseExceptions()
    # Define the paths to the aerial images
    ORTO_ROOT_FOLDER_PATH = r"/home/kitkat/Master_project/flyfoto_gaula_1963"
    # Define the path to the labels
    LABEL_ROOT_PATH = r"/home/kitkat/Master_project/labels/refined_OD_labels"
    # Define the river folders that will be processed
    RIVER_SUBFOLDER_NAMES = ["gaula_1963"]
    # Destination root path
    DEST_ROOT_PATH = r"/home/kitkat/Master_project/labels/rasters"

    # Create label rasters
    for subfolder in RIVER_SUBFOLDER_NAMES:
        # Images
        orto_folder_path = os.path.join(ORTO_ROOT_FOLDER_PATH, subfolder)
        glob_string = os.path.join(orto_folder_path, "*.tif")
        image_paths = glob.glob(glob_string)
        # Labels
        label_folder_path = os.path.join(LABEL_ROOT_PATH, subfolder)
        # Get polygons
        id_poly_dict = load_polygons(label_folder_path)
        # Create raster labels for the area covered by the image
        for path in image_paths:
            create_raster_labels(path, id_poly_dict,
                                 os.path.join(DEST_ROOT_PATH, subfolder, "label" + os.path.split(path)[-1]))
    print("Done!")

def train_valid_test_split(source_folder, dest_folder, train=0.8, valid=0.2, test=0, split_by_big_images=False):
    """
    Moves the files from a single place into train, validation and test folders.
    :param source_folder: The root folder of the data. (small images)
    :param dest_folder: The root folder where the new data will be placed.
    :param train: The size of the training set
    :param valid: The size of the validation set
    :param test: The size of the test set
    :return: Nothing
    """

    # Check that train, valid and test add up to 1
    if train + valid + test != 1:
        raise ValueError(f"train, valid and test must add up to 1. They added up to {train + valid + test}")

    os.makedirs(dest_folder, exist_ok=True)

    # Get all the path endings
    images = glob.glob(os.path.join(source_folder, "labels", "*"))
    images = [os.path.split(p)[-1] for p in images]
    if split_by_big_images:
        big_images = list(set([path[:15] for path in images]))
        tiny_images = images
        images = big_images

    random.shuffle(images)

    # Split into train val and test
    train_images = images[:int(train*len(images))]
    start_index = int(train*len(images))
    end_index = start_index + int(valid*len(images))
    val_images = images[start_index:end_index]
    start_index = end_index
    test_images = images[start_index:]

    if split_by_big_images:
        new_train_images = []
        new_val_images = []
        new_test_images = []
        for image in tiny_images:
            if image[:15] in train_images:
                new_train_images.append(image)
            elif image[:15] in val_images:
                new_val_images.append(image)
            elif image[:15] in test_images:
                new_test_images.append(image)
            else:
                raise ValueError(f"No match was found for {image}. All images should have a match.")
        train_images = new_train_images
        val_images = new_val_images
        test_images = new_test_images
    # Copy files to dest folder
    os.makedirs(os.path.join(dest_folder, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(dest_folder, "train", "images"), exist_ok=True)
    for path in train_images:
        shutil.copyfile(os.path.join(source_folder, "labels", path),
                        os.path.join(dest_folder, "train", "labels", path))
        shutil.copyfile(os.path.join(source_folder, "images", path),
                        os.path.join(dest_folder, "train", "images", path))
    os.makedirs(os.path.join(dest_folder, "val", "labels"), exist_ok=True)
    os.makedirs(os.path.join(dest_folder, "val", "images"), exist_ok=True)
    for path in val_images:
        shutil.copyfile(os.path.join(source_folder, "labels", path),
                        os.path.join(dest_folder, "val", "labels", path))
        shutil.copyfile(os.path.join(source_folder, "images", path),
                        os.path.join(dest_folder, "val", "images", path))
    os.makedirs(os.path.join(dest_folder, "test", "labels"), exist_ok=True)
    os.makedirs(os.path.join(dest_folder, "test", "images"), exist_ok=True)
    for path in test_images:
        shutil.copyfile(os.path.join(source_folder, "labels", path),
                        os.path.join(dest_folder, "test", "labels", path))
        shutil.copyfile(os.path.join(source_folder, "images", path),
                        os.path.join(dest_folder, "test", "images", path))


def experiment_7_extra_filter(poly_path, image_dir, dest_dir):
    """
    In addition to mono class removal and removal of images with too much of the unknown class this function will
    limit the data from Surna 1963 to a predefined area around the river. This area is marked by a polygon.
    :return:
    """
    os.makedirs(dest_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(image_dir, "*.tif"))
    driver = ogr.GetDriverByName("ESRI Shapefile")
    shape_ds = driver.Open(poly_path, 0)
    layer = shape_ds.GetLayer()
    polys = []
    for feature in layer:
        geom = feature.GetGeometryRef().Clone()
        ref = osr.SpatialReference()
        ref.ImportFromEPSG(25833)
        geom.AssignSpatialReference(ref)
        polys.append(geom)

    for path in image_paths:
        image_ds = gdal.Open(path)
        image_bounding_box = create_bounding_box(image_ds)
        for poly in polys:
            if image_bounding_box.Intersects(poly):
                shutil.copy(path, os.path.join(dest_dir, os.path.split(path)[-1]))
                break


def divide_and_filter_main():
    """
    Function to actually run the division and filtering process.
    :return: Nothing, it creates new files
    """
    gdal.UseExceptions()
    # Define the paths to the aerial images
    ORTO_ROOT_FOLDER_PATH = r"/media/kitkat/Seagate Expansion Drive/Master_project/new_plane_fotos"
    # Define path to label rasters
    LABEL_RASTER_ROOT_FOLDER = r"/media/kitkat/Seagate Expansion Drive/Master_project/labels/rasters"
    # Define the river folders that will be processed
    RIVER_SUBFOLDER_NAMES = ["gaula_1963", "lærdal_1976"]
    # Destination root path
    DEST_ROOT_PATH = r"/media/kitkat/Seagate Expansion Drive/Master_project/tiny_images_6"

    # Create label rasters
    label_paths = []
    image_paths = []
    for subfolder in RIVER_SUBFOLDER_NAMES:
        # Images
        label_folder_path = os.path.join(LABEL_RASTER_ROOT_FOLDER, subfolder)
        l_paths = glob.glob(os.path.join(label_folder_path, "*.tif"))
        label_paths += l_paths
        for l_path in l_paths:
            name = os.path.split(l_path)[-1].replace("label", "")
            image_path = os.path.join(ORTO_ROOT_FOLDER_PATH, subfolder, name)
            image_paths.append(image_path)
    divide_and_save_images(image_paths, label_paths, DEST_ROOT_PATH, image_size=512, do_overlap=False, do_crop=False)


def train_valid_test_split_main():
    """
    Function to run the train, valid and test split and copy function.
    :return: Nothing, creates new files
    """

    source_path = r"/media/kitkat/Seagate Expansion Drive/Master_project/tiny_images_6"
    dest_path = r"/media/kitkat/Seagate Expansion Drive/Master_project/machine_learning_dataset_7"

    train_valid_test_split(source_path, dest_path, split_by_big_images=False)


if __name__ == '__main__':
    train_valid_test_split_main()
