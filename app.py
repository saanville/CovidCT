from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a Flask app
app = Flask(__name__)

model = load_model('model.h5')
model._make_predict_function()

labels = ["COVID negative", "COVID positive"]


def model_predict(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img = np.array(img)
    img = np.resize(img, [1, 150, 150, 1])

    
    #print(img)
    res = model.predict(img)
    a = labels[int(res[0])]
    y = '' + a
    return y



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # get the image
        f = request.files['file']

        # Save the file to uploads folder
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        result = model_predict(file_path)

        return result
    return None

if __name__ == '__main__':
    app.run(port=5001)
