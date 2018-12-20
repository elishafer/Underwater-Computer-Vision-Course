
"""UW strobe simulator for HW2"""

__author__      = "Elisei Shafer"

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import cv2

def depthmap_loader(path_to_depthmap):
    """
    Load depthmap that's made in OpenSfm, clean it up and interpolate
    :param path_to_depthmap:
    :return:
    """
    depthmap = np.load(path_to_depthmap)
    plt.imshow(depthmap['depth'])
    plt.show()
    result, odm = depthmap_preprocessor(depthmap)



    return odm, result

def depthmap_preprocessor(depthmap, dfilter='interpolation'):
    dm = depthmap['depth']  # np.where( depthmap['score'] > 0.97, depthmap['depth'], 0)
    dm = cv2.medianBlur(dm, 7)

    if dfilter=='interpolation':
        mask = ~(dm == 0)
        xx, yy = np.meshgrid(np.arange(dm.shape[1]), np.arange(dm.shape[0]))
        xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T
        data0 = np.ravel(dm[mask])
        interp0 = interpolate.NearestNDInterpolator(xym, data0)
        result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)

    return result0, dm


if __name__ == '__main__':
    path_to_depthmap = '../../hw1/sfm/depthmaps/20181125_105644.jpg.clean.npz'
    (fdm, rdm)  = depthmap_loader(path_to_depthmap)

    plt.imshow(fdm)
    plt.show()
    plt.imshow(rdm)
    plt.show()
