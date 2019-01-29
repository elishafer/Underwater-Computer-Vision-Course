"""Underwater Ambient Light Simulator"""

__author__      = "Elisei Shafer"

import numpy as np
from strobe_sim import depthmap_loader, build_3d_map
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit
import pandas as pd

def simulate_ambient(J, A, depthmap, beta):
    """

    :param J:
    :param A:
    :param beta:
    :return:
    """

    t = np.exp(-beta * depthmap[:,:,None])
    I = t * img + (1 - t) * A

    return I


def get_percentile_distance(depthmap, img, p, comparison):

    percentile_value = np.percentile(img,p)
    if comparison == 'gt':
        channel_mask = np.array(img > percentile_value)
    elif comparison == 'lt':
        channel_mask = np.array(img < percentile_value)
    value_distance_array = np.array([img[channel_mask], depthmap[channel_mask]])

    return value_distance_array

def get_p_dist_all_channels(map_3d, img, p, comparison):

    red_p = get_percentile_distance(map_3d,img[:,:,0],p, comparison)
    green_p= get_percentile_distance(map_3d,img[:,:,1],p, comparison)
    blue_p = get_percentile_distance(map_3d,img[:,:,2],p, comparison)

    return [red_p, green_p, blue_p]


def create_rgbd_df(depthmap, I):

    rgbd_df = pd.DataFrame({
        "D": depthmap.flatten(),
        "R": I[:,:,0].flatten(),
        "G": I[:,:,1].flatten(),
        "B": I[:,:,2].flatten()
    })
    return rgbd_df


def find_ambient_beta_all_chan(E_d):
    ambient = np.zeros(3)
    beta = np.zeros(3)
    distance = E_d.D #E_d[0][1]
    I = [E_d.R, E_d.G, E_d.B] #[E_d[0][0], E_d[1][0], E_d[2][0]]

    for i in xrange(3):
        [popt, tmp] = curve_fit(ambient_light_irradiance, distance, I[i], bounds=(0., [1, 1]))
        ambient[i] = popt[0]
        beta[i] = popt[1]
    return [ambient, beta]


def recover_radiance(img, depthmap, ambient, beta):

    rho = 1 + (img / ambient[None, None, :] - 1) * np.exp(beta * depthmap[:, :, None])
    img_r = rho * ambient
    # img_r = rho
    img_r[img_r < 0] = 0
    img_r[img_r > 1] = 1

    # rho[:, :, 0] = rho[:, :, 0] - rho[:, :, 0].min()
    # rho[:, :, 1] = rho[:, :, 1] - rho[:, :, 1].min()
    # rho[:, :, 2] = rho[:, :, 2] - rho[:, :, 2].min()
    #
    # rho[:, :, 0] = rho[:, :, 0] / rho[:, :, 0].max()
    # rho[:, :, 1] = rho[:, :, 1] / rho[:, :, 1].max()
    # rho[:, :, 2] = rho[:, :, 2] / rho[:, :, 2].max()
    # img_r = rho

    return img_r

def ambient_light_irradiance(d, ambient_inf, beta):

    E = ambient_inf*(1 - np.exp(-beta*d))
    return E

def plot_data_curve_fit(i):
    plt.scatter(E_rws[i][1], E_rws[i][0])
    plt.plot(d, E_curve[:, 0])
    plt.show()

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
    depthmap = np.linalg.norm(map_3d, axis=2)

    img = Image.open('../20181125_105644_scaled.png')
    img = np.array(img, dtype='float64')/255.0


    # a = np.array([0.228, 0.046, 0.019])
    # b = np.array([1.22, 2.05, 3.06]) * 10 ** (-3)
    # beta = a + b
    # A = b/beta

    a = np.array([0.236, 0.068, 0.077])
    b = np.array([0.314, 0.395, 0.469])
    beta = a + b
    A = b/beta

    I =  simulate_ambient(img, A, depthmap, beta)

    plt.imshow(I)
    plt.show()

    # E = get_p_dist_all_channels(depthmap, I, 0, 'gt')
    E = create_rgbd_df(depthmap, I)
    E = E.sort_values(by=['D'])
    E = E.rolling(1000).quantile(.01)
    E = E.dropna()

    [I_inf, beta_c] = find_ambient_beta_all_chan(E)

    print(I_inf)
    print(beta_c)


    img_r = recover_radiance(I, depthmap, I_inf, beta_c)

    plt.imshow(img_r)
    plt.show()

    depthmap_rws = np.load('map_3d_rwc.npz')
    depthmap_rws = depthmap_rws['map_3d.npy']

    img_rws = Image.open('../../hw3/images/Real_world_scenes/LFT_3266_liner_undistort.tif')
    img_rws = np.array(img_rws, dtype='float64')/255.0
    # max = np.array([img_rws[:,:,i].max() for i in range(3)])
    # img_rws = img_rws/max

    # E_rws = get_p_dist_all_channels(depthmap_rws, img_rws, 50, 'lt')
    #
    # for i in xrange(3):
    #     E_rws[i] = E_rws[i][:, ~np.any(np.isnan(E_rws[i]), axis=0)]

    E = create_rgbd_df(depthmap_rws, img_rws)
    E = E.sort_values(by=['D'])
    E = E.dropna()
    E_rws = E.rolling(1000).quantile(.01)
    E_rws = E_rws.dropna()


    [I_inf, beta_c] = find_ambient_beta_all_chan(E_rws)
    img_rws_r = recover_radiance(img_rws, depthmap_rws, I_inf, beta_c)

    plt.imshow(img_rws_r)
    plt.show()


    d = np.linspace(0, 20)
    E_curve = I_inf * (1 - np.exp(-beta_c * d[:, None]))