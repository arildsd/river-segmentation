import pretrained_unet
import sys

if __name__ == '__main__':
    """
    Runs the pretrained_unet with arguments. 
    """

    model_name = sys.argv[1]
    freeze = sys.argv[2]
    image_augmentation = bool(int(sys.argv[3]))
    context_mode = bool(int(sys.argv[4]))

    train_data_folder_path = r"/home/arildsd/machine_learning_dataset_5/train"
    val_data_folder_path = r"/home/arildsd/machine_learning_dataset_5/val"
    run_path = r"/home/arildsd/runs"

    pretrained_unet.run(train_data_folder_path, val_data_folder_path, model_name=model_name, freeze=freeze,
                        image_augmentation=image_augmentation, context_mode=context_mode, run_path=run_path)
