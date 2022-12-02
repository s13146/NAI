#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Ilya Ryzhkov / Katarzyna Węsierska
# Created Date: dec 2022
# version ='0.1'
# Maszyna wektorów nośnych konstruuje hiperpłaszczyznę lub zestaw
# hiperpłaszczyzn w przestrzeni wielowymiarowej
# którą można wykorzystać do klasyfikacji, regresji lub innych zadań.
# bazy ocen filmów
# Required: numpy, pyplot, sklearn
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree, svm, datasets


def wine_calc():
    """ Obcliczenie i wyświetlenie wykresu dla wine dataset """
    wine = datasets.load_wine()
    print(wine)
    # Rozdzielamy dataset 75% traning / 25% test
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data[:, :2], wine.target, test_size=0.25, random_state=100)

    # inicjalizacja SVC
    clf = svm.SVC(kernel='rbf', C=1, gamma=1000).fit(
        X_train, y_train)

    model_tree = tree.DecisionTreeClassifier()
    model_tree.fit(X_train, y_train)

    expected_y = y_test
    predicted_y = model_tree.predict(X_test)

    print(metrics.classification_report(expected_y, predicted_y))

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    h = (x_max / x_min) / 100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    plt.subplot(1, 1, 1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(wine.data[:, :2][:, 0], wine.data[:, :2][:, 1],
                c=wine.target, cmap=plt.cm.Paired)
    plt.xlabel('Color Intensity')
    plt.ylabel('Malic Acid')
    plt.show()


if __name__ == "__main__":
    wine_calc()
