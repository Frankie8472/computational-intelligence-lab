# movie rating prediction based on given matrix entries
# 10'000 Users (rows)
# 1'000 Movies (columns)

import csv
import numpy as np
import pandas as pd

nrOfUsers = 10000
nrOfMovies = 1000

ratings = []

with open('data_train.csv', 'r') as file:
    header = file.readline()
    for line in file:
        entry, prediction = line.split(',')
        starRating = int(prediction)
        rowEntry, columnEntry = entry.split('_')
        row = int(rowEntry[1:])
        column = int(columnEntry[1:])
        ratings.append((row-1, column-1, starRating))

dataMatrix = np.zeros([nrOfUsers, nrOfMovies])

for (row, column, starRating) in ratings:
    dataMatrix[row, column] = starRating



