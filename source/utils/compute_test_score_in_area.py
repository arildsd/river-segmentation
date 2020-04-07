import gdal
import numpy as np
import os
import sys
import glob
from osgeo import ogr
from osgeo import osr
import model_utils
import data_processing
import sklearn.metrics

if __name__ == '__main__':
    # Specify file paths
    bounding_poly_path = r"/media/kitkat/Seagate Expansion Drive/Master_project/nea_1962_test_set/shapefile_corrections/test_set_boundig_area.shp"
    predicted_image_folder = r"/media/kitkat/Seagate Expansion Drive/Master_project/nea_1962_test_set/nea_1962_predictions"
    corrected_image_folder = r"/media/kitkat/Seagate Expansion Drive/Master_project/nea_1962_test_set/raster_corrections"

    # Load shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.Open(bounding_poly_path, 0)
    layer = ds.GetLayer()
    polys = []
    for feature in layer:
        geom = feature.GetGeometryRef().Clone()
        ref = osr.SpatialReference()
        ref.ImportFromEPSG(25833)
        geom.AssignSpatialReference(ref)
        polys.append(geom)

    # Find intersecting images
    predicted_image_paths = glob.glob(os.path.join(predicted_image_folder, "*.tif"))
    corrected_image_paths = glob.glob(os.path.join(corrected_image_folder, "*.tif"))

    predicted_image_overlapping_paths = []
    corrected_image_overlapping_paths = []

    for predicted_image_path in predicted_image_paths:
        # Load the image
        image_ds = gdal.Open(predicted_image_path)
        image_bounding_box = data_processing.create_bounding_box(image_ds)
        image_ds = None
        for poly in polys:
            if poly.Intersects(image_bounding_box):
                predicted_image_overlapping_paths.append(predicted_image_path)
                if os.path.split(predicted_image_path)[-1] not in [os.path.split(path)[-1] for path in corrected_image_paths]:
                    raise Exception(f"The corresponding corrected file did not exist.")
                corrected_image_overlapping_paths.append(os.path.join(corrected_image_folder, os.path.split(predicted_image_path)[-1]))
                break

    # Compute confusion matrix
    mem_driver = driver = gdal.GetDriverByName("MEM")
    prediction_arrays = []
    corrected_arrays = []
    for i in range(len(predicted_image_overlapping_paths)):
        prediction_ds = gdal.Open(predicted_image_overlapping_paths[i])
        corrected_ds = gdal.Open(corrected_image_overlapping_paths[i])

        # Make a mask with 1s at the location of pixels that should be included
        mask_ds = mem_driver.Create("", prediction_ds.RasterXSize, prediction_ds.RasterYSize)
        mask_ds.SetProjection(prediction_ds.GetProjection())
        mask_ds.SetGeoTransform(prediction_ds.GetGeoTransform())
        mask_ds.GetRasterBand(1).WriteArray(np.zeros((prediction_ds.RasterYSize, prediction_ds.RasterXSize)))
        gdal.RasterizeLayer(mask_ds, (1,), layer, burn_values=(1,))
        mask = mask_ds.GetRasterBand(1).ReadAsArray()
        mask_mean = np.mean(mask)
        prediction_array = prediction_ds.GetRasterBand(1).ReadAsArray()
        corrected_array = corrected_ds.GetRasterBand(1).ReadAsArray()

        # Flatten
        prediction_array = prediction_array.flatten()
        corrected_array = corrected_array.flatten()
        mask = mask.flatten().astype(bool)

        # Filter using the mask
        prediction_array = prediction_array[mask]
        corrected_array = corrected_array[mask]

        prediction_arrays.append(prediction_array)
        corrected_arrays.append(corrected_array)

    predictions = np.concatenate(prediction_arrays)
    corrections = np.concatenate(corrected_arrays)

    a = np.mean(predictions)
    b = np.max(predictions)
    c = np.min(predictions)

    d = np.mean(corrections)

    miou = model_utils.miou(corrections, predictions, num_classes=5)

    conf_mat = np.zeros((5, 5))
    for i in range(len(prediction_arrays)):
        conf_mat += sklearn.metrics.confusion_matrix(corrected_arrays[i], prediction_arrays[i], labels=[0, 1, 2, 3, 4])

    print(miou)
    print(np.sum([conf_mat[i][i] for i in range(5)] / np.sum(conf_mat)))
    print(conf_mat)

    np.savetxt(os.path.join("../../tests", "nea_1962_test_conf_mat.csv"), conf_mat, delimiter=",")




