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

```python
Number of training examples = 34,799
Number of validation examples = 4,410
Number of testing examples = 12,630
Image data shape = (32, 32, 3)
Number of classes = 43
```

### Data Set Visualization
Here is a histogram of all of the images that the model will be trained on. Looking at this is important because having a skewed or disproportionate data set will most likely skew the model into biased predictions.

![sign distribution](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p2-traffic-signs/traffic-sign-classifier-project/write-up/distro-signs.JPG)

I randomly chose a handful of images and their corresponding sign label from the test set. Some reasons for doing this:

* Simply get a good overview of what you have to work with
* View the resolution, color, size, darkness, etc.
* Spot any outliers or unmatched signs
* Note any features that you can take advantage of (color, shape, symbol complexity, etc.)

![visualize randomly chosen images and their sign lables](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p2-traffic-signs/traffic-sign-classifier-project/write-up/visualize-data-set.JPG?raw=true)

Design and Test a Model Architrecture
---
My model architecture of choice is the popular [LeNet-5](https://en.wikipedia.org/wiki/Convolutional_neural_network#LeNet-5) convolutional nueral network first created by Yann LeCunn et al. It's popular for working with 'small' images in that it's designed for handwritten digit classification like zipcodes or numbers in check books. This architecture appeared appropriate as traffic signs are composed of simple abstractions like sign shapes, symbols, numbers, etc.

The only preprocessing I performed on the images was to normalize each pixel value as this is a common and necessary step when implmenting gradient descent.
```python
norm_image = cv2.normalize(img, norm_img, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
```
I did not convert the images to grayscale and instead kept the images' color. As we'll see, the model appears to acknowledge the color of the image. 

The model architecture allows for a 32x32 picture with three color channels as its input. The model's output is an array of Softmax probabilities, one for each traffic sign.

Layer | Description
------|------------
Input | 32x32x3 RGB image
Convolution | Convolove the input image from 32x32x3 to 28x28x6
Activation | RELU
Max Pool | Only select salient pixels. The matrix decreases from 28x28x6 to 14x14x6
Convolution | Convolove the input image from 14x14x6 to 10x10x6
Activation | RELU
Max Pool | Only select salient pixels. The matrix decreases from 10x10x6 to 5x5x16
Flatten | Flatten the 5x5x16 matrix to an array length of 400
Fully Connect | Take input of 400 activations and output 100
Activation | RELU
Drop Out | Kill off 50% of the nuerons' activations
Fully Connect | Take input of 100 activations and output 84
Activation | RELU
Drop Out | Kill off 50% of the nuerons' activations
Fully Connect | Take input of 84 activations and output 43

### Training the model
The ultimate output of the model (after Softmax is performed on the model's Logits) is an array of predicted probablities one for each of the 43 traffic signs. Therefor the model is solving a classification problem. The calculated loss will be cross entropy which is a way to compute how 'wrong' or 'off' the model's prediction is for classificaiton scenarios. My optizer of choice is [Adaptive Moment Estimation](https://www.quora.com/Can-you-explain-basic-intuition-behind-ADAM-a-method-for-stochastic-optimization) or ADAM. Why ADAM? We don't want to use just Gradient Descent -- that would take too long. So we'll implement an approximate, but faster type called Stochastic Gradient Descent or SGD. ADAM is a type of SGD that takes advantage of its previous computed gradients in order to apply wiser, subsequent gradient calculations.

I configured my learning rate to be 0.001 -- a typical learning rate value.
My batch-size is dependent on my local computer's resources -- a batch size of a 128 images worked well.
The number of epochs I settled with is 6. That's like watching season two of House of Cards six times. Each time you watch the same season (collection of data) you may get a deeper enjoyment with each successive veiwing (a lower caculated loss / higher validation accuracy / well-adjusted weights) but watch the season too many times and you'll plateua on your derived enjoyment (stuck at a constant validation accuracy / overfitting on the training data / weights that are too atuned to the training data).

In order to acheive an acceptable accuracy from the model, I simply tuned certain hyperparameters (epoch, drop-out percentages, etc.) and retrained the network. This allow me to see how the model changes given a particular hypothesis. For example, three epochs probably cuts out much needed additional network training -- increasing the dropout percentage too much might make the model unconfident in its decision.

Analyze Testing Results
---
I split up the corpus of German traffic sign images into training, validation, and testing sets. Here's how the model performed after being trained:

Image Set | Number of Images | Overall Accuracy
----------|------------------|-----------------
Training   | 34,799 | 98.89%
Validation | 4,410 | 93.51%
Test       | 12,630 | 92.28%

Analyze Performance on novel German Traffic Signs
---
I grabbed seven german traffic signs from Google's image search. By chossing signs with graffiti I can see how this may bend the perception of the model. I also chose a sign that had an auxiliary, rectangular sign below directly below it -- an additional trait the model was not trained on.

![test images](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p2-traffic-signs/traffic-sign-classifier-project/write-up/test-images.JPG)

The model scored an overall accuracy of **71.43%** on these seven test images. Let's explore possible explanations as to why this score is much lower than the corpus' test set of **92.28%**

* The sample size is small with only seven images. That means that one image accounts for over 14 percentage points (a large step). The model classified five traffic signs correctly and two incorrectly.

* The sign **Beware of Ice/Snow** containes a relatively complex snowflake symbol. As you can see from an example sign, the network's top three prediction got the shape and outline of the sign, but the model failed to determine what exactly was inside the sign. The shape is too intricate; this might be evidence that the network architecture is not robust enough to handle complex symbols.

![Beware of Ice/Snow](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p2-traffic-signs/traffic-sign-classifier-project/write-up/test-1.JPG)

* As noted previously, I included a taffic sign that the model did indeed train on, however, as is common with traffic signs, there is an additional, smaller rectangular sign directly below it. It appears the model mistook this small, white rectanglular shape to mean **No Entry** rather than the correct **Wild Animals Crossing**.

![Wild Animals Crossing](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p2-traffic-signs/traffic-sign-classifier-project/write-up/test-3.JPG)

* I'd like to point out that the model is acute enough to detect large and small vehicle shapes within each sign. This is evident with the **No Passing for Vehicles over 3.5 Metric Tons**. The model even follows our own intuition with its next best guesses as **End of No Passing by Vehicles over 3.5 Metric Tons** and **No Passing**.

![No Passing for Vehicles over 3.4 Metric Tons](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p2-traffic-signs/traffic-sign-classifier-project/write-up/test-image-6.JPG)

* Even when an image is taken from a different perspective than perpindicular, like with **No Passing**, the model picks out the round shape, the white background, and the two car shapes. Interestingly enough, the model is slightly reluctant in its decision as it gives some possibility that the sign is **End of No Passing**.

![No Passing](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p2-traffic-signs/traffic-sign-classifier-project/write-up/test-4.JPG)

* The model appears to ignore defaced traffic signs and treats extra markings as noise.

![Graffitti](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p2-traffic-signs/traffic-sign-classifier-project/write-up/test-5.JPG)

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

Note the bottom two rows. Of these rows, the first sign, with the pedestrian and bicycle, was never in the model's training set. Yet, the model acknowledged features of this sign -- blue, round, white markings -- and still predicted a confident score for a 'Mandatory Roundabout'. This shows a bad [recall](https://en.wikipedia.org/wiki/Precision_and_recall) as the model failed to prove that it is capable of observing false negatives which it could do by giving the sign a low probability.

The last image is a picture of myself. Haven't you ever wondered what German traffic sign you are?

![all test images](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p2-traffic-signs/traffic-sign-classifier-project/write-up/all-tests.png)


### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.
