import numpy as np
import gdal
import json
import glob
import os
import sys
from osgeo import ogr
from osgeo import osr

"""
Look here for cheats:
https://pcjericks.github.io/py-gdalogr-cookbook/geometry.html
"""

def find_intersecting_polys(geometry, polys):
    intersecting_polys = []
    for poly in polys:
        if poly.Intersects(geometry):
            intersecting_polys.append(poly)
    return intersecting_polys


def rasterize_polygons(polygons, image_path, class_name, driver=gdal.GetDriverByName("GTiff")):
    """
    Retuns a data set image with the bounding box dimensions and the polygon locations marked by 1.
    :param polygons: A list of polygons of the same class
    :param bounding_box: A bounding box. type: Geometry
    :return: Data set image
    """
    # Get meta data from the image
    image_ds = gdal.Open(image_path)
    geo_transform = image_ds.GetGeoTransform()
    projection = image_ds.GetProjection()
    n_pixels_north = image_ds.RasterYSize()
    n_pixels_east = image_ds.RasterXSize()

    # Create empty image
    output_path = image_path.remove(".tif")
    output_path += f"_temp_label_{class_name}.tif"
    label_raster = driver.Create(output_path, n_pixels_east, n_pixels_north,
                  1, gdal.GetDataTypeByName("int16"))
    label_raster.SetGeoTransform(geo_transform)
    label_raster.SetProjection(projection)

    # Burn the polygons into the new image
    for poly in polys:
        gdal.Rasterize(label_raster, poly)
    # Save and close the files
    image_ds = None

    return label_raster


def create_raster_labels(image_path, polys):
    # Create bounding box for the image
    bounding_box = create_bounding_box(image_path)
    # Find intersecting polygons
    intersecting_polys = find_intersecting_polys(bounding_box, polys)
    #TODO: transform geometries to np arrays
    #TODO: figure out the post-processing on the geometries
    pass


def create_bounding_box(image_path):
    # Create a bounding box geometry for the image
    image_ds = gdal.Open(image_path)
    geo_transform = image_ds.GetGeoTransform()
    n_pixels_north = image_ds.RasterYSize()
    n_pixels_east = image_ds.RasterXSize()
    top_left_coordinate = (geo_transform[0], geo_transform[3])
    bottom_right_coordinate = (geo_transform[0] - n_pixels_north * geo_transform[1],
                               geo_transform[3] - geo_transform[5] * n_pixels_east)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(top_left_coordinate[0], top_left_coordinate[1])  # Top left
    ring.AddPoint(top_left_coordinate[0], bottom_right_coordinate[1])  # Top right
    ring.AddPoint(bottom_right_coordinate[0], bottom_right_coordinate[1])  # Bottom right
    ring.AddPoint(bottom_right_coordinate[0], top_left_coordinate[1])  # Bottom left
    rectangle = ogr.Geometry(ogr.wkbPolygon)
    rectangle.AddGeometry(ring)
    return rectangle


def load_polygons(filepath):
    polygons = []
    with open(filepath, "r") as file:
        geojson = json.load(file)
        crs = geojson["crs"]["properties"]["name"].split(":")
        if crs[0].lower() != "epsg":
            raise Exception(f"The coordinate system for {filepath} was not espg, it was {crs[0]}")
        epsg_number = int(crs[1])
        reference = osr.SpatialReference()
        reference.ImportFromEPSG(epsg_number)
        features = geojson["features"]
        for feature in features:
            geometry = feature["geometry"]
            poly = ogr.CreateGeometryFromJson(str(geometry))
            poly.AssignSpatialReference(reference)
            polygons.append(poly)
    return polygons


if __name__ == '__main__':
    gdal.UseExceptions()

    PLANE_IMAGES_FOLDER = r"data/plane_images"
    GEOJSON_PATH = r"../data/polygons/gaula_1963_river.geojson"
    polys = load_polygons(GEOJSON_PATH)
    test_image_path = "../data/test/gaula_1963.tif"
    create_raster_labels(test_image_path, polys)
    # Take all the geotifs and intersect them with the polygons.
    plane_images_file_paths = glob.glob(os.path.join(PLANE_IMAGES_FOLDER, "*", "*.tif"))
    for image_path in plane_images_file_paths:
        # Intersect label polygons and create a raster label
        #create_raster_labels()
        pass