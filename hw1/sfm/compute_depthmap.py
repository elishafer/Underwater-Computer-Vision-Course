#!/usr/bin/env python

"""compute_depthmap.py: For UW CV homework 1"""

__author__      = "Elisei Shafer"

import sys
sys.path.append('../calibration')

import math
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

from project_to_2d import project_to_2d


with open('undistorted_reconstruction.json') as f:
    recon_data = json.load(f)

points = []
x_s = []
y_s = []
z_s = []

for point_number in recon_data[0]['points']:
    points.append(
        recon_data[0]['points'][point_number]['coordinates'])

for point in points:
    x_s.append(point[2])
    y_s.append(point[1])
    z_s.append(point[1])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x_s, y_s, z_s)
# plt.show()


# image_name = '20181125_105606.jpg'
image_name = '20181125_105631.jpg'
# image_name = '20181125_105648.jpg'


#Get rotation and translation data from json file
img_2d = []
camera_translation = recon_data[0]['shots'][image_name]['translation']
camera_rotation = np.array(recon_data[0]['shots'][image_name]['rotation'])

#convert rotation vector to rotation matrix:
rotation_matrix = cv2.Rodrigues(camera_rotation)[0]
#Create Rt matrix
Rt = np.empty((3, 4))
Rt[:, :3] = rotation_matrix
Rt[:, 3] = camera_translation

image_point_list = []
distances_list =[]
w = 3264
h = 2448


#Project points to image plane results in:
# image_point_list - points on the image plane
# distances_list - distance of each point away from camera.
for e,point in enumerate(points):
    point.append(1)
    point_b = Rt * np.transpose(np.matrix(point))
    if point_b[2] < 0:
        continue
    image_point =project_to_2d(point_b , distortion=False)
    distance = math.sqrt(sum([(point[i] - camera_translation[i]) ** 2
                   for i in xrange(3)]))
    if distance > 60:
        continue
    if abs(image_point[0]) <  w and abs(image_point[1]) < h:
        image_point_list.append(image_point)
        distances_list.append(distance)


img = np.zeros((h + 1, w + 1))
# extract x,y coordinates:
x, y = [point[0] for point in image_point_list], [point[0] for point in image_point_list]

# Set size of dot to see map better:
dot_size = 15
# Place points in image:
for e, point in enumerate(image_point_list):
    for i in xrange(dot_size+1):
        if int(point[1])+dot_size-i > h:
            continue
        for j in xrange(dot_size+1):
            if int(point[0])+dot_size-j > w:
                continue
            img[int(point[1])+dot_size-i, int(point[0])+dot_size-j] = distances_list[e]


plt.imshow(img,interpolation='nearest')
plt.colorbar()
plt.show()
