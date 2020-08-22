#Flask libraries
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import tensorflow as tf
import keras

#Keras libraries
from keras.models import load_model
from keras.preprocessing import image

#Define a Flask app
app = Flask(__name__)

MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)