import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2
import pydot


def load_dataset(directory):
    images = []
    labels = []
    for idx, label in enumerate(uniq_labels):
        for file in os.listdir(directory+'/'+label):
            filepath = directory + '/' +label +'/' +file
            img = cv2.resize(cv2.imread(filepath), (50, 50))
            images.append(img)
            labels.append(idx)
    images = np.asarray(images)
    labels = np.asarray(labels)
    return images, labels


def display_images(x_data, y_data, title, display_label = True):
    x, y = x_data, y_data
    fig, axes = plt.subplots(5, 8, figsize = (18, 5))
    fig.subplots_adjust(hspace= 0.5, wspace=0.5)
    fig.suptitle(title, fontsize=18)
    for i, ax in enumerate(axes.flat):
        ax.imshow(cv2.cvtColor(x[i], cv2.COLOR_BGR2RGB))
        if display_label:
            ax.set_xlabel(uniq_labels[y[i]])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

#loading_dataset into X_pre and Y_pre
data_dir = r'D:/miniproject/111SignRecognition/train'
uniq_labels = sorted(os.listdir(data_dir))
X_pre, Y_pre = load_dataset(data_dir)
print(X_pre.shape, Y_pre.shape)

#spliting dataset into 80% train, 10% validation and 10% test data
X_train, X_test, Y_train, Y_test = train_test_split(X_pre, Y_pre, test_size = 0.8)
X_test, X_eval, Y_test, Y_eval = train_test_split(X_test, Y_test, test_size = 0.5)


#matplotlib notebook

#print shapes and show example for each set
print("Train images shape", X_train.shape, Y_train.shape)
print("Test images shape", X_test.shape, Y_test.shape)
print("Evaluate image shape", X_eval.shape, Y_eval.shape)
display_images(X_train, Y_train, 'Samples from Train Set')
display_images(X_test, Y_test, 'Samples from Test set')
display_images(X_eval, Y_eval, 'Samples from validation set')

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
Y_eval = to_categorical(Y_eval)
X_train = X_train / 255
X_test = X_test/255
X_eval = X_eval/255

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape=(50, 50, 3)),
    tf.keras.layers.Conv2D(16, (3,3), activation = 'relu'),
    tf.keras.layers.Conv2D(16, (3,3), activation= 'relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
    ])

model.summary()

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_eval, Y_eval))

#testing
model.evaluate(X_test, Y_test)

model.save(r'D:/miniproject/111SignRecognition/sample_train.h5')

train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']


#plotting training and validation loss vs. epoch

# epochs = list(range(1, 19))
# plt.plot(epochs, train_loss, label = "training loss")
# plt.plot(epochs, val_loss, label = "validation loss")
# plt.legend()
# plt.show()
#
# plt.plot(epochs, train_acc, label = "training accuracy")
# plt.plot(epochs, val_accuracy, label = "validation loss")
# plt.legend()
# plt.show()

print('completed...')
