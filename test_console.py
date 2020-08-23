from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np

model = load_model('model.h5')

img = cv2.imread('uploads/1.png')
img = cv2.resize(img, (150, 150))
img = np.resize(img, [1, 150, 150, 1])

result = model.predict(img)
print(result[0])