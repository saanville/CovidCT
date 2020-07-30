#importing the libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Dense
import tensorflow as tf
from glob import glob
import pandas as pd
import numpy as np
import cv2
import os


#Tweak according to your needs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1.89GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1900)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

#Setting the parameters
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
DIR = 'Dataset/'


#Image augmentation
training_data = ImageDataGenerator(rescale=1./255,
                                   validation_split=0.2,
                                   horizontal_flip=True,
                                   rotation_range=5,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   shear_range=0.05,
                                   zoom_range=0.05,)

#Training data generator
train_generator = training_data.flow_from_directory(DIR,
                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                  batch_size=BATCH_SIZE,
                                                  color_mode="grayscale",
                                                  class_mode='binary',
                                                  subset="training")

#Validation data generator
validation_generator = training_data.flow_from_directory(DIR,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         batch_size=BATCH_SIZE,
                                                         class_mode='binary',
                                                         color_mode="grayscale",
                                                         subset='validation')


model = Sequential()

model.add(Conv2D(64,(3, 3), activation='relu', padding="same", input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(128,(3, 3), activation='relu', padding="same"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(128,(3, 3), activation='relu', padding="same"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='RMSProp', metrics=['accuracy'])

history = model.fit_generator(train_generator,
                              verbose=1,
                              validation_data = validation_generator,
                              epochs=50)

#loss: 0.1285 - accuracy: 0.9565 - val_loss: 1.3400 - val_accuracy: 0.5811 using adam
#loss: 0.1956 - accuracy: 0.9197 - val_loss: 3.4767 - val_accuracy: 0.6486 using RMSProp