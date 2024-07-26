import numpy as np


def read_keypoints(filename):
    # read keypoints from ref_kpts.dat file
    # 12 keypoints in ZXY order
    num_keypoints = 12
    fin = open(filename, 'r')

    kpts = []
    while (True):
        line = fin.readline()
        if line == '': break

        line = line.split()
        line = [float(s) for s in line]

        line = np.reshape(line, (num_keypoints, -1))
        kpts.append(line)

    kpts = np.array(kpts)
    return kpts