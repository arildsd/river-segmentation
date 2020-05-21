import data_processing

if __name__ == '__main__':
    shapefile_path = r"/media/kitkat/Seagate Expansion Drive/Master_project/"
    image_dir = r"/media/kitkat/Seagate Expansion Drive/Master_project/tiny_images_5/labels"
    dest_dir = r"/media/kitkat/Seagate Expansion Drive/Master_project/tiny_images_6/labels"

    data_processing.experiment_7_extra_filter(shapefile_path, image_dir, dest_dir)
