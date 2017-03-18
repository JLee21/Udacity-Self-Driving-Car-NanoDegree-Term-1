import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout

# If using AWS server, set this to True to change the file data's path
aws = True

if aws:
    csv_path = "Udacity-Self-Driving-Car-Nanodegree\p3-bahavioural-cloning\
    carnd-behavioral-cloning-p3\data\data\driving_log.csv"
    image_path = "Udacity-Self-Driving-Car-Nanodegree\p3-bahavioural-cloning\
    carnd-behavioral-cloning-p3\data\data\IMG\\"
else:
    csv_path = "D:\SDC\p3-Behavioural-Cloning\data\data\driving_log.csv"
    image_path = "D:\SDC\p3-Behavioural-Cloning\data\data\IMG\\"

line = []
with open(csv_path) as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in tqdm(lines):
    source_path = line[0]
    filename = source_path.split('/')[-1]
    image_path_list = image_path + filename
    image = cv2.imread(image_path_list)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

"""
Flip Image and Steering Angle
"""
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

"""
Plot
"""
plt.hist(augmented_measurements, 50)
plt.show()

'''
Model
'''
batch_size = 128

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# barebones model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Convolution2D(nb_row=5, nb_col=5, nb_filter=16))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
# model.add(Dense(84))
# model.add(Activation('relu'))
model.add(Dense(1))

# model = Sequential()
# model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# model.add(Convolution2D(nb_row=3, nb_col=3, nb_filter=32))
# model.add(MaxPooling2D(strides=None))
# model.add(Dropout(0.5))
# model.add(Activation('relu'))
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(84))
# model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=1, batch_size=batch_size)

model.save('model-v1.h5')
