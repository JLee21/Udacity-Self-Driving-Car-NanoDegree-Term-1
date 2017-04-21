## Project: Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
![gif]()

The goal of this project is to detect and track vehicles on a road using a combination of machine learning and computer vision techniques.

Report Outline
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

I chose the YCrCb image space since this space proved best among other image spaces when testing. From the graph above, the model takes advantage of the spread of values in the `Y` or luminance channel as well as the tight luminance groupings; the black car's luminance values are grouped low while the white car's values are grouped high on the specturm. Additionally, the roadside vegetation green colors and the yellow lane lines, for example, are respectively grouped together.
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

Collecting the RGB pixel values is implementing in cell block `Functions`, lines `21-26` and the RGB histogram values are shown in lines `28-38`.

### Support Vector Machine Model
As mentioned in the section directly before, a feature vector is collected from each sampling of a test-image. To reiterate, this feature vector contains the values of the HOG output, RGB pixel values, and the histogram output of the RGB values. The length of this vector is 4140. A Support Vector Machine model was chosen as this type of model is well suited for this type of car/not-car classification challenge since a SVM creates the starkest decision boundary within a reasonable training time of 8 seconds over two thousand images.

The training data was a collection of car and non-car images, about one thousand for each.

![]()

The feature vector is exctracted from each of the training/test images and a Standard Scaler was applied. In order to efficeintly train the SVM model, this sacler functions alters the columns of each feature vector so that the column-wise mean is zero with a unit variance. The code for scaling the trainging/test images is in cell block `Create Model`, lines `31-34`.

Given that SVM models have numerous hyperparameters to select, the Python library Sklearn's GridSearchCV was used to test out the following combination of paramters.

![svm-params]()

The best paramters in regards to the best test score and training time follows:
Paramter | Value
---------|------
Kernel | `RBF`
C | `10`
gamma | `0.0001`


## Sliding Window Search
In order to determine if a vehicle is present in an image, a Sliding Window Search technique was used. Rather than searching the entire image at once for the prescense of a vehicle, a computaionally costly step, a small, fixed-sized window slides across the image extracting the pixel values. Furthermore, three different window scales were used and each window scale only searched parts of the image, as shown below:

![window scale]()

The image below helps illustrate a window sliding across a subsection of the original image where each grid color represents a different window scale and corresponding image subsection.

![exampel grid search]()

The scaling of the window was selected based on various testing. I found that small window scales (less than 0.7) produced spurious false positives while large window scales (more than 1.2) appeared to overwhelm the overall vehicle detection space by creating large overlaps into non-vehicle space.

As a window slides ove its grid space, it extracts the feature vector that correspond to within the window's boundaries. To reiterate the feature vector, the composition of the feature vector consits of the RGB pixel values, the RGB histogram values, and the HOG output. In particular, to help expedite the HOG implementation, the HOG output is performed only once for each image. The sliding window then subsamples parts of the whole image HOG output as it traverses its grid space.

An important paramter selection was how much the window slides across the image everytime it peforms a new feature vector extraction. This value was determined empirically to be 32 pixels. If the window traversed just 16 pixels, the overall speed of the feature vector extraction was reduced considerably without any noticible imporvement in accurate vehicle detections. If the window travserved 64 pixels, the generated heatmap appeared to be too sparse and at times intepreted a sinble vehilce as two or three spearate vehicles.

Once the feature vector is extracted, the SVM model tests if there is a vehicle. If the model predicts a vehicle, that window space is added to a heatmap and an accumulation of positive vehicle detections begins to form as shown below:

![]()

The code for calling multiple featuve vector extractions with various windows sized can be found in code block `Heatmaps` in the function `heatmap`. The feature vector extraction takes place in code block `HOG/Pixel Subsamplilng Function` in the function `find_cars`.

### Pipeline Optimizations
In order to filter out false positive vehicle detections, a threshold was applied to each heatmap. This threshold dropped parts of the produced heatmap that accrued only one positive vehicle detection. The implmentation is located in cell block `Pipeline Helper Functions`, lines `1-4` and is called from cell block `Heatmaps`, lines `21-22`.

I noticed a *Bermuda Triangle* of loss vehicle detections at a medium range distance within the driving footage. As a vehilce would pass about 30-50m ahead, its detection would slowly drop. I implemented a more spatially narrow, small-scaled window serach in this area as shown as a red grid below:

![subsample grid space]()

## Video Implementation
A vehicle tracking class `VehicleTracker` was created to organize the all the functions needed in the processing pipeline. The class is created in cell block `Create Vehicle Tracker Class`. At a basic level, a single class is created to handle each image frame that is supplied from a driving video. The image is passed through processing functions that extract any and all feature vectors from the image and applies the findings to a heatmap. As heatmaps are collected from each image, the heatmap is averaged over the last 20 stored heatmaps. This not only produced smoother heatmap drawings and less jittery bounding boxes, but it also helped ignore single vehilce detections by relying on a concentration of detections. I chose to store the last 20 heatmaps because this retained enough historic detections from surrounding vehicles so as to not miss (low frame count) or lag fast moving vehilces (high frame count).a

![avg heatmaps]()

In order to group a concentration of vehicle detections into a single bounding box, SciPy's `label()` function was used as this helped determine the boundaries of a collection of pixels as shown below:

![label-examle]()

The code for heatmap labeling is located in cell block `Create Vehicle Tracker Class`, lines `26-27`.

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

