import pretrained_unet
import sys

if __name__ == '__main__':
    """
    Runs the pretrained_unet with arguments using a data generator that flows from a directory. 
    """

    model_name = sys.argv[1]
    freeze = sys.argv[2]

    train_data_folder_path = r"/home/arildsd/machine_learning_dataset_6_png/train"
    val_data_folder_path = r"/home/arildsd/machine_learning_dataset_6/val"
    run_path = r"/home/arildsd/runs"

    pretrained_unet.run_from_dir(train_data_folder_path, val_data_folder_path, model_name=model_name, freeze=freeze,
                                 run_path=run_path)
