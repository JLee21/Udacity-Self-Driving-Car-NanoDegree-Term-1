## Project: Implement Behaviorial Cloning in a Car Simulator
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

Project Outline
---
* Files Submitted 
* Camera Calibration
* Image Pipeline
* Video Pipeline
* Discussion

## Files Submitted
* `advanced-lane-finding.ipynb` - the notebook contains all of the code used for advanced lane finding. All code referenced pertains to this notebook unless otherwise noted
* `write-up-images` - this directory contains all of the example output images used in this writeup
* `project-video.mp4` - final output video showcasing advanced lane finding techniques

## Camera Calibration

The camera that is located in the front of the car needs to be undistorted. This steps invovles taking several calibartion images of a printed 2D chessboard with the same camera. The location of the calibration images used are stored in this repository's directory `cal-images`. A function from OpenCV is used to find the corners of the chessboard. 
```python
# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
```
These corner vertices are compared to a plane mesh grid of points that are evenly seprated on flat plane (shown as `Mesh Grid of objpoints` below). 
![](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p4-advanced-lane-finding/carnd-advanced-lane-lines/write-up-images/objpoints-draw-corners.JPG)
The OpenCV function `calibrateCamera` computes these points and outputs a corrected camera matrix `mtx` and distortion coefficients `dist`.
```python
# Perform camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size ,None,None)
```
Below is an example of a distortion corrected calibration image.

![](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p4-advanced-lane-finding/carnd-advanced-lane-lines/write-up-images/cam-cal-undistort.JPG)


## Image Pipeline

### Distortion Correction
Every image must undergo a undistortion step as preceding steps require or assume no distortion from the inherent nature of the camera lenses is induced. In order to undistort each image the following code is used (cell block ` `)

Below is an example of distorion corrected image that is part of the driving video.

![]()

### Color Transforms - Gradients

### Perspective Transorm
In order to compute a second order polynomial fit of the left and right lane lines, a birds-eye-view of the lane must be made. This is completed by using the OpenCV's funciton apply perspective. The source points `src` and destination points `dst` are hard-coded and used throughtout the entirety of the video. These four vertices instruct the function how much to warp an image so that it appears as if we're looking directly down on it.
The code can be found in the cell block titled ` ` as well as directly below
```python 

```


### Polynomial Fit of Lane Lines

After finding the left and right pixels that denote the prescence of a line the numpy function `polyfit` is used to fit a second-order polynomial function to those respective lane line pixels.
```python
np.polyfit(lefty, leftx, 2)
np.polyfit(righty, rightx, 2)
```

Below are a few examples where the left and right lane line pixels are colored red and blue, respectively. The yellow lines are the second-order polynomial fit. When fitting the polynomial function, only the respective lane-colored pixels are used.

![]()

![]()

### Radious of Curvature


### Vehicle's Lane Position

Like in the previous section `Radius of Curvature`, the pixel dimension of the image was converted to meters.

### Final Plot of Lane Space and Statistics

![]()

![]()

## Video Pipeline

The final video submission, `video.mp4`, was created using a python module `MoviePy`. The code below (cell block `Full Test Clip - Used for Project Submission`) takes the function `apply_lane_find` (cell block `Full Image Pipeline`) and applies it to each of the video's image frame.

```python
from moviepy.editor import VideoFileClip
clip1 = VideoFileClip(project-video.mp4')
proj_clip = clip1.fl_image(apply_lane_find)
```

## Discussion
