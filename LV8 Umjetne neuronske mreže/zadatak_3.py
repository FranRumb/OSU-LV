import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


SeqModel = keras.models.load_model('SeqModel.keras')

testImage = Image.open("test.jpg").convert("L")


testImage = np.expand_dims(testImage, -1)
testImage = testImage.astype(float)/255
testImage = testImage.reshape(1,28,28,1)

prediction = SeqModel.predict(testImage)
pred_class = np.argmax(prediction, axis=1)

print(pred_class)