#Flask libraries
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#TensorFlow libraries
import tensorflow as tf
import keras

#Keras libraries
from keras.models import load_model
from keras.preprocessing import image

#Other libraries
import cv2
import numpy as np

#Define a Flask app
app = Flask(__name__)

MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)

labels = ["Non COVID", "COVID"]

img = cv2.imread('Positive/2.png')
img = cv2.resize(img, (150, 150))
img = np.resize(img, [1, 150, 150, 1])
result = model.predict(img)

labels[int(result[0])]