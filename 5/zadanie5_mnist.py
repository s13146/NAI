#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Ilya Ryzhkov / Katarzyna Węsierska
# Created Date: dec 2022
# version ='0.1'
# Algorytm buduje i uczy sieć neuronową na podstawie zbioru fashion_mnist
# Required: numpy, tensorflow, matplotlib
# ---------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np


def mnist_nn():
    """ Główna funkcja nauczania i budowy sieci neuronowej """
    # zbior danych
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()

    # sieć neuronową
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # nauczanie
    model.fit(train_images, train_labels, epochs=5)

    # testowanie
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test accuracy:', test_acc)

    # wykres predykcji
    predictions = model.predict(test_images)

    # confussion matrix
    cm = tf.math.confusion_matrix(
        labels=test_labels, predictions=np.argmax(predictions, axis=1)).numpy()

    # wyświetlanie confussion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confussion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()


if __name__ == "__main__":
    mnist_nn()
