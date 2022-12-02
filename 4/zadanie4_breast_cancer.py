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
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree, svm, datasets


def breast_cancer_calc():
    """ Obcliczenie i wyświetlenie wykresu SVC dla breast_cancer dataset """
    cancer = datasets.load_breast_cancer()

    # Rozdzielamy dataset 70% traning / 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.3, random_state=109)

    h = .02  # mesh step
    C = 1.0  # Regularisation
    clf = svm.SVC(kernel='rbf', C=C, gamma=100).fit(
        X_train[:, :2], y_train)

    model_tree = tree.DecisionTreeClassifier()
    model_tree.fit(X_train, y_train)

    expected_y = y_test
    predicted_y = model_tree.predict(X_test)

    print(metrics.classification_report(expected_y, predicted_y))

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    scat = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    legend1 = plt.legend(*scat.legend_elements(),
                         loc="upper right", title="diagnostic")
    plt.xlabel('mean_radius')
    plt.ylabel('mean_texture')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


if __name__ == "__main__":
    breast_cancer_calc()
