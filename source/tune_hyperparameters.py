import numpy as np
import os
import sys
import model
import copy
from datetime import date

def tune_parameters(train_set_X, train_set_y, valid_set_X, valid_set_y, param_dict, logfile_path):

    hist = model.run(train_set_X, train_set_y, depth=param_dict["depth"], kernel_size=param_dict["kernel_size"],
                     number_of_convolutions=param_dict["number_of_convolutions"], filters=param_dict["filters"],
                     activation=param_dict["activation"], momentum=param_dict["momentum"],
                     learning_rate=param_dict["learning_rate"], drop_rate=param_dict["drop_rate"],
                     do_validate=True, do_test=False,
                     valid_set_X=valid_set_X, valid_set_y=valid_set_y, logfile=logfile_path)


if __name__ == '__main__':
    POINTER_FILE_PATH = r"D:\pointers\05"

    # Train set
    train_set_X, train_set_y = model.load_dataset(os.path.join(POINTER_FILE_PATH, "train.txt"))
    train_set_X = model.image_augmentation(train_set_X)
    train_set_y = model.image_augmentation(train_set_y)
    valid_set_X, valid_set_y = model.load_dataset(os.path.join(POINTER_FILE_PATH, "valid.txt"))

    logfile_path = "../results"

    param_dict_1 = {"depth": 3, "kernel_size": 5, "number_of_convolutions": 3, "filters":32, "activation": "relu",
                    "momentum": 0.0, "learning_rate": 0.001, "drop_rate": 0.0}
    param_dict_2 = copy.deepcopy(param_dict_1)
    param_dict_2["depth"] = 4
    param_dict_3 = copy.deepcopy(param_dict_1)
    param_dict_3["depth"] = 4
    param_dict_3["drop_rate"] = 0.5
    param_dict_4 = copy.deepcopy(param_dict_1)
    param_dict_4["drop_rate"] = 0.5
    param_dict_5 = copy.deepcopy(param_dict_1)
    param_dict_5["filters"] = 16
    param_dict_6 = copy.deepcopy(param_dict_1)
    param_dict_6["depth"] = 5
    for i, param_dict in enumerate([param_dict_1, param_dict_2, param_dict_3, param_dict_4, param_dict_5, param_dict_6]):
        tune_parameters(train_set_X, train_set_y, valid_set_X, valid_set_y, param_dict,
                        os.path.join(logfile_path, str(date.today()) + "_" + str(i) + ".log"))