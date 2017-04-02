## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

Data Set Summary & Exploration
---
I wanted to understand the data that I was working with -- the shape, the values, etc. I went ahead and established the shapes of the data and even graphed the bare bones of the values.
I also grabbed the bigger-picture part of the dataset, like how much I'm working with.

![data set summary](https://github.com/JLee21/Udacity-Self-Driving-Car-NanoDegree/blob/master/p2-traffic-signs/traffic-sign-classifier-project/write-up/basic-data-set-summary.JPG)

### Data Set Visualization
I randomly chose a handful of images and their corrsponding sign label from the test set. Some reasons for doing this is to observe traights about the data set so you can prepare for the best and the worst.
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

In order to acheive acceptable accuracy from the model, I simply tuned certain hyperparameters (epoch, drop-out percentages, etc.) and retrained the network. This allow me to see how the model changes given my hypothesis. For example, three epochs probably cuts out needed 

### Testing Results
I split up the corpus of German traffic sign images into training, validation, and testing sets. Here's how the model performed:
Image Set | Overall Accuracy
----------|----------
Training | 98.89%
Validation | 93.51%
Test | 92.28%



### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. [Download the dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which we've already resized the images to 32x32.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.
