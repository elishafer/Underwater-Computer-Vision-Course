# Principles in UW Imaging - HW 1
## Part 1 - Intrinsic calibration:
###a. Calibrate camera intrinsic params:
The code for this is located in `camera_calibration.py`

Checkerboard photos are located in folder `checkerboard`.

The code basically takes the images in the checkerboard folder and looks for a
checkerboard pattern. Points are saved and and plugged into the camera calibration
function. The code was adapted from the openCV tutorial.

In the end the output for the calibration matrix was the following:
```
[[  2.56055184e+03   0.00000000e+00   1.53353035e+03]
 [  0.00000000e+00   2.56023518e+03   1.20626158e+03]
 [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]
```
Distortion Coefficients are `(k1,k2,p1,p2,k3)` :
```
[[ 0.28558109 -1.57311711 -0.01224433 -0.02108002  2.45055228]]
```
###b. Object coordinates
Code was written calculate the projection of object at point `(1,1,1)` and then
at point `(1,1,10)`.

The code for the calculation is in `project_to_2d.py`. In the code we first take
the x and y coordinates and divide each of them by z. Next, we distort the image using
the barrel and pincusion distortion models and the coefficients found earlier.
After that the calibration matrix is multiplied by distorted x,y and z=1.
The coordinates found were the following:
```
('2d projection of coordinate 1 %:\n', (1, 1, 1))
[[ 39363.89981736]
 [ 39077.19548992]]
('2d projection of coordinate 2 %:\n', (10, 1, 1))
[[  6.42388262e+10]
 [  6.42308661e+09]]
('2d projection of coordinate 3 %:\n', (1, 1, 10))
[[ 1788.10582301]
 [ 1461.25799252]]
 ```
The first and second calculations will be out of the frame of the camera. The third
will be in the frame.

The distortion coefficients are probably incorrect. This may be because the
checkerboard pattern wasn't straight but wavy.

## Part II - SfM
###a. Taking the photos
20 photos were taken using a cellphone camera (Samsung Galaxy A7 2017) from different angles.

###b. SfM
For this part I used [OpenSfm](https://github.com/mapillary/OpenSfM). Open SfM takes in
images from the and processes them for SfM. 

![Mesh](sfm.gif)

###c and d - generate distance map
OpenSfM generates depthmaps and stores them as .npz files. The files were opened in python
and the depthmap found and plotted. Values for the colorbar are in meters.
![depthmap1](sfm/105635.png)
![depthmap2](sfm/105648.png)

