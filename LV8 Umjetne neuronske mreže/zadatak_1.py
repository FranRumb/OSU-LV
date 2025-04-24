import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)
labels = ['1', '2', '3', '4',  '5', '6', '7', '8', '9', '10']
# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa
plt.figure()
plt.imshow(x_train[0])
plt.show()
plt.imshow(x_train[1])
plt.show()
plt.imshow(x_train[2])
plt.show()

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
SeqModel = keras.Sequential()
SeqModel.add(layers.Input(shape=input_shape))
SeqModel.add(layers.Dense(100, activation='relu'))
SeqModel.add(layers.Dense(50, activation='relu'))
SeqModel.add(layers.Flatten())
SeqModel.add(layers.Dense(10, activation='softmax'))
SeqModel.summary()


# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
SeqModel.compile(loss=keras.losses.categorical_crossentropy, 
                 optimizer=keras.optimizers.Adam(), 
                 metrics=['accuracy'])

# TODO: provedi ucenje mreze
#history = SeqModel.fit(x_train_s, y_train_s, batch_size=128, epochs=9, verbose=1, validation_split=0.1)
#SeqModel.save('SeqModel.keras')
SeqModel = keras.models.load_model('SeqModel.keras')
prediction = SeqModel.predict(x_test_s)
y_class_pred= np.argmax(prediction, axis=1)
y_class_true = np.argmax(y_test_s, axis=1)

# TODO: Prikazi test accuracy i matricu zabune
print(SeqModel.evaluate(x_test_s, y_test_s))
print(y_class_true[0])
print(y_class_pred[0])
print(y_class_true[1])
print(y_class_pred[1])
cm = confusion_matrix(y_true=y_class_true, y_pred=y_class_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# TODO: spremi model

#SeqModel.save('SeqModel.keras')