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
###b. Object coordinates
Code was written calculate the projection of object at point `(1,1,1)` and then
at point `(10,1,1)`.

The code for the calculation is in `project_to_2d.py`. The code speaks for itself.
Results are the following:
```
('2d projection of coordinate 1 %:\n', (1, 1, 1))
[[  4.09408220e+03]
 [  3.76649676e+03]
 [  1.00000000e+00]]
('2d projection of coordinate 2 %:\n', (10, 1, 1))
[[  2.71390488e+04]
 [  3.76649676e+03]
 [  1.00000000e+00]]
 ```