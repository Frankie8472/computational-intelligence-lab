#!/usr/bin/env python3
import data_util as util
import fill_matrix
import improve

users = 10000
movies = 1000
data_path = '../input_data/data_train.csv'

# Fill unknown values
A, ratings = fill_matrix.fill(data_path, users, movies)
output_path = '../output_data/step1.csv'
util.create_submission_csv(A, output_path)

# Improve
improve.run(A, ratings, users, movies)
# output_path = '../output_data/step2.csv'


