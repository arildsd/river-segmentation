import numpy as np
import os
import sys
import model
import copy
from datetime import date


def tune_parameters(train_set_X, train_set_y, valid_set_X, valid_set_y, param_dict, logfile_path, iterations=5):
    scores = []
    for i in range(iterations):
        logfile_path = logfile_path.replace(".log", f"_{i}.log")
        hist = model.run(train_set_X, train_set_y, depth=param_dict["depth"], kernel_size=param_dict["kernel_size"],
                         number_of_convolutions=param_dict["number_of_convolutions"], filters=param_dict["filters"],
                         activation=param_dict["activation"], momentum=param_dict["momentum"],
                         learning_rate=param_dict["learning_rate"], drop_rate=param_dict["drop_rate"],
                         do_validate=True, do_test=False,
                         valid_set_X=valid_set_X, valid_set_y=valid_set_y, logfile=logfile_path)
        val_miou = max(hist.history["val_sparse_Mean_IOU"])
        scores.append(val_miou)
    median_score = np.median(scores)
    print("Parameters:", param_dict)
    print(f"Median val miou: {median_score}")
    print(f"Max val miou: {max(scores)}")
    print(f"Min val miou: {min(scores)}")
    print("\n")

    with open(logfile_path, "a+") as f:
        f.write(f"\nMedian val miou: {median_score}\n")
        f.write(f"Max val miou: {max(scores)}\n")
        f.write(f"Min val miou: {min(scores)}")


if __name__ == '__main__':
    POINTER_FILE_PATH = r"D:\pointers\02"

    # Train set
    train_set_X, train_set_y = model.load_dataset(os.path.join(POINTER_FILE_PATH, "train.txt"))
    #train_set_X = model.image_augmentation(train_set_X)
    #train_set_y = model.image_augmentation(train_set_y)
    valid_set_X, valid_set_y = model.load_dataset(os.path.join(POINTER_FILE_PATH, "valid.txt"))

    logfile_path = "../results"

    param_dict_1 = {"depth": 3, "kernel_size": 5, "number_of_convolutions": 3, "filters": 16, "activation": "relu",
                    "momentum": 0.0, "learning_rate": 0.01, "drop_rate": 0.0}



    tune_parameters(train_set_X, train_set_y, valid_set_X, valid_set_y, param_dict_1,
                        os.path.join(logfile_path, str(date.today()) + ".log"))