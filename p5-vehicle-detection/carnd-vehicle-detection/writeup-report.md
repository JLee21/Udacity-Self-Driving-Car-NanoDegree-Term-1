## Project: Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
The goal of this project is to detect and track vehicles on a road using a combination of machine learning and computer vision techniques.

Project Outline
---
* Files Submitted
* Histogram of Oriented Gradients (HOG)
* Sliding Window Search
* Video Implementation
* Discussion
---

## Files Submitted
* `vehilce-detection.ipnb` - the notebook contains all of the code used for vehicle detection. All code referenced pertains to this notebook
* `output-images` -  a directory that holds all supporting images used in this writeup-report
* `video.mp4` - the final output video displaying the use of vehicle detection
* `writeup-report.md` - a markdown report outling and justifying all rubric points of the project.

## Histogram of Oriented Gradients (HOG)
This section explores what features were selected and why as well as what machine learning model was ultimately used for vehicle detection.
### Feature Vector Selection
By taking a look at sample test image, we can easily pick out the two cars, however, the algorithm only 'sees' pixel values. 

![test-image]()

In order to help distinguis a group of pixel values that represent a car vesus those that denotes a tree, for example, the image space has been seperated into YCrCb values, shown below.

![]()

I took the YCrCb channel values and calculated the Histogram of Orientated Gradients (HOG) for each channel. The purpose of this comes from the utility of HOG in determining an object within an image. HOG groups pixels into cells `pix_per_cell` and then each cell into blocks `cell_per_block`. The gradient of the given pixel values are calculated within each cell and the gradient is grouped into `orient` number of bins when taking the histogram.
I found the following values to be fine enough to reliably detect cars and the absence of cars from test images.

HOG Parameter | Value
--------------|-------
`pix_per_cell`| 16
`cell_per_block`|2
`orient`        |9

The image grid below shows the contrasts of a car that has gone under HOG versus an image that does not have a car. The values I chose for the HOG function reliably highlight the signatures of a car such as where the hood of the car begins or the stark contrast between the ground and the bottom of the car. When compared to the HOG output of a non-car image, it is clear that the HOG parameters are able to grasp the overall shape of the car and even the stark change in gradient from the car's taillight or license plate. A non-car HOG output may show edges, however, it does not contain or exhibit the overall rectangular shape of the car.

![]()

The code for implementing HOG is shown in cell block `Functions`, lines `1-19`.

In addition to collecting HOB values as part of the feature vector, the RGB pixel values of an image are added in two ways: by collecting the pixel values of an image and concatnecting them to the feature vector and also collecting the histogram of each RGB channel and concatenacting those values to the feature vector. When looking at the RGB color space for a given test image, the color of the cars tend to be vibrant colors when contrasted with the surround road, scenary, etc.

![test-image]()

![rgb-image-space]()

### Support Vector Machine Model

## Sliding Window Search


## Video Implementation


## Discussion




**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

