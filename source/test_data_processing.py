import unittest
import data_processing
import numpy as np
import gdal
import json
import glob
import os
import sys
from osgeo import ogr
from osgeo import osr


class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        self.shapefile_folder_path = r"..\testing_data\labels"
        self.orto_folder_path = r"..\testing_data\orto"
        self.image_path = r"..\testing_data\orto\33-2-462-209-13.tif"
        self.test_image_ds = gdal.Open(self.image_path)

    def test_poly_load(self):
        id_poly_dict = data_processing.load_polygons(self.shapefile_folder_path)
        # Test that expected keys (ids) are present
        for key in id_poly_dict.keys():
            self.assertTrue(key in [0, 1, 2, 3, 4, 5])
        for key in range(0, 5):
            self.assertTrue(key in id_poly_dict.keys())
        # Check that the geometries actually are geometries
        for poly_list in id_poly_dict.values():
            for poly in poly_list:
                self.assertIsNotNone(poly)
                self.assertIsInstance(poly, ogr.Geometry)

    def test_create_bounding_box(self):
        resolution = 0.2  # meters
        bounding_box = data_processing.create_bounding_box(self.test_image_ds)
        # Check that the bounding box is a geometry object
        self.assertIsInstance(bounding_box, ogr.Geometry)
        # Check that the area of the image matches that of the bounding box (within a tolerance)
        area_of_image = self.test_image_ds.RasterYSize*resolution * \
                        self.test_image_ds.RasterXSize*resolution
        area_of_box = bounding_box.Area()
        self.assertAlmostEqual(area_of_box, area_of_image, delta=0.05*min(area_of_image, area_of_box),
                               msg=f"Image area: {area_of_image}\tBox area: {area_of_box}")





if __name__ == '__main__':
    unittest.main()