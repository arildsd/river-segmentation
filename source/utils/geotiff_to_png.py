import faulthandler; faulthandler.enable()
import os
import gdal
import numpy as np
import sys
import glob
from PIL import Image

"""
Convert geotiffs to pngs. The georeferences will be lost. 
"""


if __name__ == '__main__':
    source_dir = sys.argv[1]
    dest_dir = sys.argv[2]

    tif_paths = glob.glob(os.path.join(source_dir, "*.tif"))
    for tif_path in tif_paths:
        ds = gdal.Open(tif_path)
        band_arrays = []
        for band in range(ds.RasterCount):
            band_array = ds.GetRasterBand(band + 1).ReadAsArray()
            band_array = np.expand_dims(band_array, axis=-1)
            band_arrays.append(band_array)
        if ds.RasterCount > 1:
            image_array = np.concatenate(band_arrays, axis=-1).astype(np.uint8)
        else:
            image_array = band_array.squeeze(axis=-1)
        im = Image.fromarray(image_array)
        im.save(os.path.join(dest_dir, os.path.split(tif_path)[-1]).replace(".tif", ".png"))
