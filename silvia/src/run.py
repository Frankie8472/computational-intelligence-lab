#!/usr/bin/env python3
import fill_matrix

users = 10000
movies = 1000
data_path = '../input_data/data_train.csv'
output_path = '../output_data/step1.csv'

cross_validate = False

# Fill unknown values
fill_matrix.fill(data_path, users, movies, output_path, cross_validate)

