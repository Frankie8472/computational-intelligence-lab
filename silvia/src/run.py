#!/usr/bin/env python3
import numpy as np

import data_util as util
import fill_matrix
import improve

users = 10000
movies = 1000
data_path = '../input_data/data_train.csv'

runs = 5
A_best = None
score_best = 100

for i in range(runs):
    print('Run ' + str(i+1) + ' of ' + str(runs) + '..........')
    # Init matrix
    A, ratings = util.csv_to_rating_matrix(data_path, np.nan, users, movies)
    asked_entries = set()

    A, asked_entries = util.split_data(A, ratings, np.full((movies, 1),
        np.nan), True)

    # Init unknown ratings with values
    A = fill_matrix.fill(A, ratings, data_path, users, movies)

    score = util.score(A, asked_entries)
    print(score)

    # Improve
    A = improve.run(A, ratings, asked_entries, users, movies, 1)
    score = util.score(A, asked_entries)
    print(score)
    if score < score_best:
        A_best = A
        score_best = score

# Write to output file
output_path = '../output_data/svd_k13_' + str(score_best) + '.csv'
util.create_submission_csv(A_best, output_path)

