## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
![]()
The goal is of this project is to construct a Convolutional Nueral Network to classify German Traffic signs.

This ReadMe outlines the step of the project
* Data Set Summary & Exploration
* Design and Test a Model Architecture
* Analyze Testing Results
* Analyze Performance on novel German Traffic Signs

Data Set Summary & Exploration
---
I wanted to understand the data that I was working with -- the shape, the values, etc. I went ahead and established the shapes of the data and even graphed the bare bones of the values.
I also grabbed the bigger-picture part of the dataset, like how much I'm working with.

![data set summary](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p2-traffic-signs/traffic-sign-classifier-project/write-up/basic-data-set-summary.JPG)

### Data Set Visualization
I randomly chose a handful of images and their corresponding sign label from the test set. Some reasons for doing this is to observe traights about the data set so you can prepare for the best and the worst.
* Simply get a good overview of what you have to work with
* View the resolution, color, size, darkness, etc.
* Spot any outliers or unmatched signs

![visualize randomly chosen images and their sign lables](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p2-traffic-signs/traffic-sign-classifier-project/write-up/visualize-data-set.JPG?raw=true)

Design and Test a Model Architrecture
---
My model architecture of choice is the popular [LeNet-5](https://en.wikipedia.org/wiki/Convolutional_neural_network#LeNet-5) convolutional nueral network first created by Yann LeCunn et al. It's popular for working with 'small' images in that it's designed for handwritten digit classification like zipcodes or numbers on checks. This architecture appeared appropriate as traffic signs are composed of simple abstractions like sign shape, symbols, numbers, etc.

The only preprocessing I performed on the images was to normalize each pixel value as this is a common and necessary step when implmenting gradient descent.
```python
norm_image = cv2.normalize(img, norm_img, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
```
I did not convert the images to grayscale and instead left the color in the images. As we'll see, the model appears to acknowledge the color of the image. 

The model architecture allows for a 32x32 picture with three color channels as its input. The output is a Softmax probablitiy of what the model computes as its decision.

Layer | Description
------|------------
Input | 32x32x3 RGB image
Convolution | Convolove the input image from 32x32x3 to 28x28x6
Activation | RELU
Max Pool | Only select salient pixels. The matrix decreases from 28x28x6 to 14x14x6
Convolution | Convolove the input image from 14x14x6 to 10x10x6
Activation | RELU
Max Pool | Only select salient pixels. The matrix decreases from 10x10x6 to 5x5x16
Flatten | Flatten the 5x5x16 matrix to an array with length of 400
Fully Connect | Take input of 400 activations and output 100
Activation | RELU
Drop Out | Kill off 50% of the nueron's activations
Fully Connect | Take input of 100 activations and output 84
Activation | RELU
Drop Out | Kill off 50% of the nueron's activations
Fully Connect | Take input of 84 activations and output 43

### Training the model
The problem to solve is a classification one. The model will output its predicted probablities for each of the 43 traffic signs. So, the calculated loss will be cross entropy. My optizer of choice was is [Adaptive Moment Estimation](https://www.quora.com/Can-you-explain-basic-intuition-behind-ADAM-a-method-for-stochastic-optimization) or ADAM. Why ADAM? We don't want to use just Gradient Descent -- that would take too long. So we'll implement an approximate, but faster type called Stochastic Gradient Descent (SGD). ADAM is a type of SGD that takes advantage of its previous computed gradients in order to be wiser for its next gradient calculation. 
I configured my learning rate to be 0.001 -- a typical learning rate value.
My batch-size is dependent on my local computer resources -- a size of 128 worked well.
The number of epochs I settled with is 6. That's like watching season one of House of Cards six times. Each time you watch same season (collection of data) and you may get deeper enjoyment with each successive veiwing (a lower caculated loss / higher validation accuracy) but watch it too many times over you'll plateua on the derived enjoyment from the show (stuck at a constant validation accruacy or even overfit on the training data).

In order to acheive acceptable accuracy from the model, I simply tuned certain hyperparameters (epoch, drop-out percentages, etc.) and retrained the network. This allow me to see how the model changes given my hypothesis. For example, three epochs probably cuts out much needed additional training for the network.

Analyze Testing Results
---
I split up the corpus of German traffic sign images into training, validation, and testing sets. Here's how the model performed:

Image Set | Overall Accuracy
----------|----------
Training | 98.89%
Validation | 93.51%
Test | 92.28%

Analyze Performance on novel German Traffic Signs
---
I grabbed 7 german traffic signs from Google's image search. By chossing signs with grafitti (or graphics -- Pink Panther) I can see how this may bend the perception of the model. I also chose a sign that had an auxiliary, rectangular sign below -- something the model was not trained on.

![test images](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p2-traffic-signs/traffic-sign-classifier-project/write-up/test-images.JPG)

The model scored an overall accuracy of 71.43% on these seven test images. Let's explore possible explanations as to why this score is much lower than the corpus' test set of 92.28%

* The sample size is small with only seven images. That means that one image accounts for over 14 percentage points. The model classified five traffic signs correctly and two incorrectly.

* The sign 'Beware of Ice/Snow' containes a relatively complex snowflake symbol. As you can see from an example sign, the network's top three prediction got the shape and outline of the sign, but the model failed to determine what exactly what the complicated black symbol was inside the shape.
![]()

* As noted previously, I included a taffic sign that the model did indeed train on, however, as is common with traffic signs, there is an additional, smaller rectangular sign directly below it. It appears the model mistook this small, white rectanglular shape to mean 'No Entry' rather than the correct 'Wild Animals Crossing'

![]()

* I'd like to point out that the model is acute enough to detect large and small vehicle shapes within each sign. This is evident with the 'No Passing for Vehicles over 3.4 Metric Tons'. The model even follows on our own intuition with its next best guesses as 'End of No Passing by Vehicles over 3.5 Metric Tons' and 'No Passing'.

![]()

* Even when the test image skews the sign 'No Passing', the model picks out the round shape, the white background, and the two car shapes. Interestingly enough, the model is slightly reluctant in its decision as it gives some possibility that the sign is 'End of No Passing'

![]()

* The model appears to ignore defacings to traffic signs and treats extra markings as noise.

![]()

Here are the rest of the model's predictions. The layout goes like this:

```pyython
--------------------------------------------------------
<test image 1> | <1st prediction> | <2nd preditionc> ...
--------------------------------------------------------
<test image 2> | <1st prediction> | <2nd preditionc> ...
--------------------------------------------------------
<test image 3> | <1st prediction> | <2nd preditionc> ...
--------------------------------------------------------
.
.
.
```

Note the bottom two rows. Of these rows, the first sign, with the pedestrian and bicycle, was never in the training set. Yet, the model took traights of this sign -- blue, round, white markings -- and still predicted a confident score that it is a 'Mandatory Roundabout'.
The last image is a picture of myself. Haven't you ever wondered what German traffic sign you are?

![]()


### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.
