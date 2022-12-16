#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Ilya Ryzhkov / Katarzyna Węsierska
# Created Date: dec 2022
# version ='0.1'
# Algorytm buduje i uczy sieć neuronową na podstawie zbioru wine
# Required: sklearn, tensorflow
# ---------------------------------------------------------------------------
from tensorflow.keras import Sequential, layers
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def wine_nn():
    """ Główna funkcja nauczania i budowy sieci neuronowej """
    # ladowanie i dzielenie zbioru danych
    X, y = load_wine(return_X_y=True)
    y = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.3)

    input_dim = X_train.shape[1:]
    output_dim = y.shape[1]

    # inicjalizacja sieci i trenowanie
    model = Sequential()
    model.add(layers.Dense(units=50, activation='relu', input_shape=input_dim))
    model.add(layers.Dense(units=50, activation='relu'))
    model.add(layers.Dense(units=50, activation='relu'))
    model.add(layers.Dense(units=50, activation='relu'))
    model.add(layers.Dense(units=50, activation='relu'))
    model.add(layers.Dense(output_dim, activation='softmax'))

    # categorical_crossentropy do klasyfikacji wieloklasowej
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(
        X_valid, y_valid), epochs=1000, verbose=0)

    print('Test accuracy:', model.evaluate(X_test, y_test)[1])


if __name__ == "__main__":
    wine_nn()
