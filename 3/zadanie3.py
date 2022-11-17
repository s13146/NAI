#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Ilya Ryzhkov / Katarzyna Węsierska
# Created Date: nov 2022
# version ='0.1'
# Algorytm oblicza Euclidean distance i generuje rekomendacje na podstawie
# bazy ocen filmów
# Required: numpy
# ---------------------------------------------------------------------------
import json
import numpy as np


def euclidean_score(dataset, user1, user2):
    common_movies = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    if len(common_movies) == 0:
        return 0

    squared_diff = []

    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_diff.append(
                np.square(dataset[user1][item] - dataset[user2][item]))

    return 1 / (1 + np.sqrt(np.sum(squared_diff)))


def find_intervals(data, user):
    min_value = 1
    person = None
    for key, value in data.items():
        if key != user:
            _i = euclidean_score(data, user, key)
            if min_value > _i:
                min_value = _i
                person = key

    return person


def get_recomendations(data, user):
    _fav = find_intervals(data, user)
    for key, value in data[_fav].items():
        if key not in data[user]:
            result[key] = value
    result_sort = list()
    for k, v in sorted(result.items(), key=lambda item: item[1]):
        result_sort.append({k: v})
    print("5 not recomended films: ")
    for item in result_sort[:5]:
        print(list(item.keys())[0])
    print("5 recomended films: ")
    for item in result_sort[-5:]:
        print(list(item.keys())[0])


if __name__ == '__main__':
    ratings_file = 'ratings.json'
    result = {}
    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())

    print("========== Test for Paweł Czapiewski =============")
    get_recomendations(data, "Paweł Czapiewski")

    print("========== Test for Wanda Bojanowska =============")
    get_recomendations(data, "Wanda Bojanowska")

    print("========== Test for Szymon Olkiewicz =============")
    get_recomendations(data, "Szymon Olkiewicz")
