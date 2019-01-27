"""Underwater Ambient Light Simulator"""

__author__      = "Elisei Shafer"

import numpy as np
from strobe_sim import depthmap_loader, build_3d_map
import cv2
from matplotlib import pyplot as plt
from PIL import Image

def simulate_ambient(J, A, map_3d, beta):
    """

    :param J:
    :param A:
    :param beta:
    :return:
    """
    Z = np.linalg.norm(map_3d, axis=2)
    t = np.exp(-beta * Z[:,:,None])
    I = t * img + (1 - t) * A

    return I

if __name__ == '__main__':
    path_to_depthmap = '../../hw1/sfm/depthmaps/20181125_105644.jpg.clean.npz'
    (fdm, rdm)  = depthmap_loader(path_to_depthmap, image_resize=False)
    image_size = (640, 480)
    # image_size = (3264, 2448)

    # plt.imshow(fdm)
    # plt.colorbar()
    # plt.show()
    # plt.imshow(rdm)
    # plt.colorbar()
    # plt.show()

    map_3d = np.load('map_3d_1mp.npz')
    map_3d = map_3d['map_3d']/2.0

    img = Image.open('../20181125_105644_scaled.png')
    img = np.array(img, dtype='float64')/255.0


    a = np.array([0.228, 0.046, 0.019])
    b = np.array([1.22, 2.05, 3.06]) * 10 ** (-3)
    beta = a + b
    A = b/beta

    # a = np.array([0.236, 0.068, 0.077])
    # b = np.array([0.314, 0.395, 0.469])
    # beta = a + b
    # A = b/beta

    I =  simulate_ambient(img, A, map_3d, beta)

    plt.imshow(I)
    plt.show()
