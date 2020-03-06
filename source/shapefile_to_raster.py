from data_processing import burn_labels_to_image
import glob
import os

"""
Burn a shapefile onto a set of raster images. Warning: This will change the images, not make copies.
Designed to be used with the EPSG:25833 georeference system.
Class ids: 0: Water. 1: Gravel. 2:Vegetation. 3:Farmland. 4: Human-constructions. 5: Unknown
"""

if __name__ == '__main__':
    image_folder = r"/media/kitkat/Seagate Expansion Drive/Master_project/dataset_extention/surna_1963_predictions/working_on"
    shapefile_path = r"/media/kitkat/Seagate Expansion Drive/Master_project/dataset_extention/surna_1963_predictions/working_on/human-constructions.shp"
    class_id = 4
    print(os.path.exists(shapefile_path))
    image_paths = glob.glob(os.path.join(image_folder, "*.tif"))
    for path in image_paths:
        burn_labels_to_image(path, shapefile_path, class_id)

