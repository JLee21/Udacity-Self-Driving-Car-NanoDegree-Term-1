## Project: Implement Behaviorial Cloning in a Car Simulator
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
![](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p3-behavioural-cloning/carnd-behavioral-cloning-p3/write-up/video.gif)

Project Outline
---
* Files Submitted 
* Collect Driving Data and Image Processing
* Build a Deep Neural Network
* Train and Validate
* Test the Model
---
[//]: # (Image References)
[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Files Submitted
#### Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* The main notebook - ai-model-notebook.ipynb - contains the code to create and train the model. All code referenced pertains to this notebook unless otherwise noted
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup-report.md summarizing the results

#### To run the car autonomously
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### Submission code is usable and readable
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Data Collection  

The car records what it sees with a virtual camera at the front of the vehicle. Along with the car's corresponding steering angle, this is what the data collection consists of.

![](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p3-behavioural-cloning/carnd-behavioral-cloning-p3/write-up/original-image.JPG)

The charts below show the distribution of all the data collected. Note that no image processing/augmentation has taken place.

![](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p3-behavioural-cloning/carnd-behavioral-cloning-p3/write-up/steering-angle-distribution.png)

I recruited a friend to drive the car around the track as close to perfection as possible. One clockwise and one counterclockwise. He also included extreme turn corrections because the car needed to learn how to recover its deviation from the center of the road.

![](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p3-behavioural-cloning/carnd-behavioral-cloning-p3/write-up/extreme-turns.JPG)

To avoid the car from being biased toward turning left or right, additional image/steering angle data was added by using OpenCV's flip method. The steering angle also had to be flipped as well
```python
cv2.flip(img, 1)
angle * -1.0
```
![](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p3-behavioural-cloning/carnd-behavioral-cloning-p3/write-up/image-flip.JPG)

To further train the car to return to the center of the lane, I took adavantage of the left and right cameras on the car. Essentially, I treated the car's left camera as its center-facing camera and changed the steering angle by an offset. That way, if the car center camera sees a picture similar to its left camera image, it will be trained to return to the lane's center.

![](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p3-behavioural-cloning/carnd-behavioral-cloning-p3/write-up/steer-offsets.JPG)

For each image that goes into training the model enters an image pipeline (located in the notebook's cells `Create Generator Helper Functions`

Before image preprocessing, each image starts off as a Red, Green, and Blue channeld image (160x320x3)

![](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p3-behavioural-cloning/carnd-behavioral-cloning-p3/write-up/original-image.JPG)

The image is then converted to a Hue Saturation Value (HSV) image and the Saturation channel is extracted. Exctracting just the S-channel helped the car detect the boundaries of the road greatly because it allows the edges of the road to standout while the road itself appears as a mostly flat color.

![](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p3-behavioural-cloning/carnd-behavioral-cloning-p3/write-up/hsv-image.JPG)

![](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p3-behavioural-cloning/carnd-behavioral-cloning-p3/write-up/only-saturation-image.JPG)

By now, the shape of the image is (160x320x1), where there is only one channel and not three (3 -> 1). The model simply does not need those original spatial dimensions (160x320) even though they help us humans see what the car sees. A image size of 64x64 works well for training as this retains enough data to train (not to mention the huge amount of training time saved!). In addition to resizing the image, the model doesn't need to be trained on the sky (top of the image) nor the hood of the car (bottom of the image) so let's crop those parts.
```python
rows, cols = 64, 64
cv2.resize(img, (rows, cols), cv2.INTER_AREA)

# cropping is performed within a Keras layer Cropping2D as mentioned later
```
![](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p3-behavioural-cloning/carnd-behavioral-cloning-p3/write-up/resized-imgae.PNG)
![](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p3-behavioural-cloning/carnd-behavioral-cloning-p3/write-up/cropped-final-image.png)

Train and Validate Split
---
A train/validate data split of 20% was implemented:
```
python train_samples, validation_samples = train_test_split(lines, test_size=0.2)
```

Python Generator
---
A python generator was created for a couple of reasons. 
* We don't want to load thousands of images all at once into the computer's RAM. Instead the images are loaded in small batches from a SSD (I found a SSD has about a 50% speed increase compared to a HD).
* We can image process these small batches of images when it's needed by the model
The generator was created in the cell block `Create Generator` and the generator is implemented in the cell block `Train, Validate, and Save Model`. In addition, a generator for the validation images was used as well.
```python
 model.fit_generator(generator=train_generator ... validation_data=validation_generator)
```
During training, the model is trained on one small batch of images at a time. Here's the distribution of steering angles of five randomly chosen batches.
![](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p3-behavioural-cloning/carnd-behavioral-cloning-p3/write-up/batch-distribution.png)
Unlike the original, unprocessed distribution of steering angles as shown earlier, the distributuion of each batch better resembles a normal Gaussian distribution. This is important as this enables the car to be robust in accepting multiple images with differnt corresponding steering angles. In other words, we want the car to drive straight in the middle of the road -- most of the time -- but we also want the car to know what to do during sharp turns, gradual turns, center-of-the-lane offsets, etc.

## Build a Deep Nerual Network
My model is constructed in the cell **Construct Model** within `ai-model-notebook`.
The model follows the following structure:

Layer | Description
------|------------
Input / Cropping | Accept a input image with a resized shape of (64,64,1). Also, crop the top 21% and bottom 8% of the image
Lambda | Normalize the pixel values of each image from a range of 0-255 to +/- 0.5
Convolution | Convolve image into 32 feature maps using a 5x5 kernel size
Activation | RELU
Max Pool | Only select salient pixels with a kernel size of 2x2
Convolution | Convolve previous layer into 64 feauture maps using a 5x5 kernel size
Activation | RELU
Max Pool | Only select salient pixels with a kernel size of 2x2
Convolution | Convolve previous layer into 128 feauture maps using a 3x3 kernel size
Activation | RELU
Max Pool | Only select salient pixels with a kernel size of 2x2
Drop Out | Kill off 20% of the previous neurons' activations
Flatten | Take the previous layer's activation and flatten the values into an array
Fully Connect | Take input of the Flatten layer and link to 512 neurons
Activation | RELU
Drop Out | Kill off 10% of the previous neurons' activations
Fully Connect | Take input of the Flatten layer and link to 100 neurons
Activation | RELU
Fully Connect | Take input of the previous layer and link to 10 neurons
Activation | RELU
Fully Connect | Take input of the previous layer and link to 1 neuron

Input
---

The input of the model is a resized image from the generator. Resizing the image greatly increases training time without giving up detail for training. A few student blogs recommended 64x64 pixels [here](http://ottonello.gitlab.io/selfdriving/nanodegree/2017/02/09/behavioral_cloning.html) and [here](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9)
```python
model.add(Cropping2D(cropping=((14,5),(0,0)), input_shape=(64, 64, 1)))
```
The model also crops away the top and bottom of the image -the sky and the car's hood- as these are parts of the image are unrelated to the steering angle.
Input normalization is important for all neural networks to allow for a successful and effecient gradient descent--this model is no exception--a Lambda layer normalizes all pixel values from a range of 0-255 to +/- 0.5
```python
model.add(Lambda(lambda x: x / 255.0 - 0.5))
```

Activation
---
The model includes Rectified Linear Unit (ReLU) Activation layers to introduce nonlinearity. This type of activation ignores negative inputs and is shown to be computationally efficient for deep neural networks.

Convolution / MaxPool
---
[](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
I implemented a similar architecture to NVidia's end-to-end model. Similar to other student's architecture, [here]() and [here]() as well as NVidia's, a common theme to extract more and more feature layers with each subsequent convolutional layer. The resoning is that each convolution layer extracts higher and higher levels of abstractions from the previous convolution layer.
This is why I the depth of each of my convolution layers are 32, 64, 128. You can see below that the each convolution layer get higher and higher in abstraction.

![](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p3-behavioural-cloning/carnd-behavioral-cloning-p3/write-up/conv-layer-1.png)

![](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p3-behavioural-cloning/carnd-behavioral-cloning-p3/write-up/conv-layer-2.png)

![](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p3-behavioural-cloning/carnd-behavioral-cloning-p3/write-up/conv-layer-3.png)

Fully Connected Layers
---
After testing various implementations of the number of fully connected layers and the number of connections each one hold, I ened up using four fully connected layers. The model concludes with a single output neruon that denotes a steering angle, as this driving challenge is a regression one, not classification. I found that the number of neurons made no significant impact on the performance of the car, although increasing the number of neurons also increased training time. A general rule of thumb is to quickly descend the number of neurons in each layer: 512 -> 100 -> 50 -> 10 -> 1

Drop Out Layers
---
I added two drop out layers that only drop a small percentage of the previous layers' activations.
```python
model.add(Dropout(0.2))
```
Before adding the dropout layers, I noticed the car would become 'stickier' to certain parts of the course -- as if it memorized exactly what it wanted to do. If I added too much droppage, the car seemed to be more 'slippery' in that it seemed to refuse to stick on a particular path and would drift, especially on the curves. I ended up only dropping 20% of the third convolution layer's activation and only 10% of the first fully connected layer's activations.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Model Optimizer and Loss Calculator
---
I chose a ADAM optimizerr (code cell titled `Train, Validate, and Save Model`). A learning rate does not need to be implemented as this is built into the optimizer. ADAM is a type of SGD that takes advantage of its previous computed gradients in order to apply wiser, subsequent gradient calculations.
```python
model.compile(loss='mse', optimizer='adam')
```
Much like solving a simple regression problem: 

![http://pgfplots.net/tikz/examples/regression-residuals/](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p3-behavioural-cloning/carnd-behavioral-cloning-p3/write-up/regression-example.JPG)

[image source](http://pgfplots.net/tikz/examples/regression-residuals/)

the model's loss is calculated using Mean Squared Error loss. This was chosen as the model tries to *fit* not *classify* a steering angle to its input image. 

## Train and Validate

Training Strategy
---
I implemented a piece of advice from my previous project review in that the model's training is conditional on its improvement; it stops training when the error loss stops decreasing. I average the last three validation loss and compared that value with the current validation loss -- if the current one is less than the average loss continue training! In addition, the model is saved after each epoch, that is, only if the validation loss improves.

![](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p3-behavioural-cloning/carnd-behavioral-cloning-p3/write-up/model-mean-squared-error-loss.png)

I found the absolute value of the training or validation mean squared error loss was not an explicit indicator that the car would drive successfully.


