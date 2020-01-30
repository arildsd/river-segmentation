import pretrained_unet
import sys

if __name__ == '__main__':

    model_name = sys.argv[1]
    freeze = sys.argv[2]
    image_augmentation = bool(int(sys.argv[3]))

    train_data_folder_path = r"/media/kitkat/Seagate Expansion Drive/Master_project/machine_learning_dataset/train"
    val_data_folder_path = r"/media/kitkat/Seagate Expansion Drive/Master_project/machine_learning_dataset/val"

    pretrained_unet.run(train_data_folder_path, val_data_folder_path, model_name=model_name, freeze=freeze,
                        image_augmentation=image_augmentation)
