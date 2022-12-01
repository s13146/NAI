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
from sklearn import svm, datasets


def breast_cancer_calc():
    """ Obcliczenie i wyświetlenie wykresu SVC dla breast_cancer dataset """
    breast_cancer = datasets.load_breast_cancer()
    print(breast_cancer)
    X = breast_cancer.data[:, :2]  # filtrowanie
    y = breast_cancer.target
    # inicjalizacja SVC
    svc = svm.SVC(kernel='rbf', C=1, gamma=100).fit(X, y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max / x_min) / 100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.subplot(1, 1, 1)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVC with linear kernel')
    plt.show()


if __name__ == "__main__":
    breast_cancer_calc()
