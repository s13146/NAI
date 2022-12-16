#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Ilya Ryzhkov / Katarzyna Węsierska
# Created Date: dec 2022
# version ='0.1'
# Algorytm buduje i uczy sieć neuronową na podstawie zbioru imdb
# Required: numpy, tensorflow, matplotlib
# ---------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from keras.datasets import imdb


def imdb_nn():
    """ Główna funkcja nauczania i budowy sieci neuronowej """
    # pobieranie danych
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

    # przygotowanie danych
    x_train = keras.preprocessing.sequence.pad_sequences(x_train,
                                                         value=0,
                                                         padding='post',
                                                         maxlen=256)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=256)

    # budowanie modelu
    model = keras.Sequential()
    model.add(keras.layers.Embedding(10000, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # trenowanie
    model.fit(x_train, y_train, epochs=40, batch_size=512,
              validation_data=(x_test, y_test), verbose=1)

    # testowanie
    results = model.evaluate(x_test, y_test)

    print('Test accuracy:', results[1])


if __name__ == "__main__":
    imdb_nn()
