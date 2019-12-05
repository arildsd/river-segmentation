import numpy as np

def conf_mat_to_latex(conf_mat):
    for i in range(len(conf_mat)):
        conf_mat[i] = [str(e) for e in conf_mat[i]]
    strings = [" & ".join(array) for array in conf_mat]
    return "\\\\ \n".join(strings)


if __name__ == '__main__':
    conf_mat = [[0, 0, 1256719, 0, 0, 0],
                [0, 0, 245976, 0, 0, 0],
                [0, 0, 12013128, 0, 0, 0],
                [0, 0, 4596505, 0, 0, 0],
                [0, 0, 651839, 0, 0, 0],
                [0, 0, 1355385, 0, 0, 0]]
    print(conf_mat_to_latex(conf_mat))