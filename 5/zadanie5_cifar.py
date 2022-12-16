#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Ilya Ryzhkov / Katarzyna Węsierska
# Created Date: dec 2022
# version ='0.1'
# Algorytm buduje i uczy sieć neuronową na podstawie zbioru cifar10
# Required: tensorflow
# ---------------------------------------------------------------------------
import tensorflow as tf


def cifar_nn():
    """ Główna funkcja nauczania i budowy sieci neuronowej """
    # załadowanie CIFAR10
    cifar10 = tf.keras.datasets.cifar10

    # Ustawienie wielkości danych wejściowych i wyjściowych
    n_outputs = 10  # 10 klas zwierząt

    # Rozdzielenie danych na zbiór treningowy i testowy, wraz z normalizacją
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Stworzenie modelu sieci neuronowej
    model = tf.keras.models.Sequential()

    # Dodanie warstw do modelu
    model.add(tf.keras.layers.Flatten(input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(n_outputs, activation='softmax'))

    # Kompilacja modelu
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Trenowanie modelu
    model.fit(x_train, y_train, epochs=5)

    # Ocena modelu na danych testowych
    test_acc = model.evaluate(x_test, y_test, verbose=2)

    print('Test accuracy:', test_acc[1])


if __name__ == "__main__":
    cifar_nn()
