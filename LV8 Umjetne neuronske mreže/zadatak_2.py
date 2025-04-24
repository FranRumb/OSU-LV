import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
num_classes = 10

x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

SeqModel = keras.models.load_model('SeqModel.keras')
prediction = SeqModel.predict(x_test)

y_class_pred= np.argmax(prediction, axis=1)
y_class_true = np.argmax(y_test_s, axis=1)

wrongClass = []

for i in range(len(y_class_pred)):
    if y_class_pred[i] != y_class_true[i]:
        wrongClass = np.append(wrongClass, i)

plt.figure()

for i in range(len(wrongClass)):
    plt.imshow(x_test[wrongClass[i].astype(int)])
    plt.title("Predicted: " + str(y_class_pred[wrongClass[i].astype(int)]) + ", Actual: " + str(y_class_true[wrongClass[i].astype(int)]))
    plt.show()