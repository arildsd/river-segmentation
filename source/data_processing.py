import numpy as np
import gdal
import json
import glob
import os

"""
Look here for cheats:
https://pcjericks.github.io/py-gdalogr-cookbook/geometry.html
"""

def find_intersecting_polys(geometry, polys):



def create_raster_labels(image, polys):
    # Find intersecting polygons
    intersecting_polys = find_intersecting_polys(image, polys)


if __name__ == '__main__':
    PLANE_IMAGES_FOLDER = r"data/plane_images"
    POLY_FOLDER = r"data/polys"
    # Take all the geotifs and intersect them with the polygons.
    plane_images_file_paths = glob.glob(os.path.join(PLANE_IMAGES_FOLDER, "*", "*.tiff"))
    for image_path in plane_images_file_paths:
        # Intersect label polygons and create a raster label
        create_raster_labels()
