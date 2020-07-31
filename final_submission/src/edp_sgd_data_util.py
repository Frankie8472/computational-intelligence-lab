import numpy as np
import random
import pandas as pd
from typing import *


# return a list of entries that should be predicted
def get_asked_entries() -> List[Tuple[int, int]]:
    '''
    Return a list of entries that should be predicted.

    :returns: List of tuples (r, c) for each asked entry, where r is the row
        index and c i the column index of the asked entry
    '''
    asked_entries = []
    with open('../input/sampleSubmission.csv', 'r') as sample_file:
        sample_file.readline()  # throw the header away
        for line in sample_file:
            entry, prediction = line.split(',')
            row_entry, column_entry = entry.split('_')
            row = int(row_entry[1:])
            column = int(column_entry[1:])
            asked_entries.append((row-1, column-1))
    return asked_entries


def store_data_float(result_matrix: 'numpy.ndarray', file_name: str) -> None:
    '''
    Write submission file.

    :param result_matrix: prediction matrix
    :param file_name: path to output file
    '''
    file = open(file_name, 'w+')  # open a new file to write into
    file.write('Id,Prediction\n')  # the header line
    asked_entries = get_asked_entries()
    for (i, j) in asked_entries:
        file.write('r' + str(i+1) + '_c' + str(j+1) + ',' +
                str(result_matrix[i][j]) + '\n')  # store with the same ID as the sample


def parsef(line: str) -> Tuple[int, int, int]:
    '''
    Parse a line in input data/submission format.

    :param line: string of the form rD_cD,F, where D are integers and F a
        float
    :returns: extracted row, column and rating values
    '''
    l1 = line.decode('utf8').split(',')
    l2 = l1[0].split('_')
    row = int(l2[0][1:])
    column = int(l2[1][1:])
    value = float(l1[1])
    return row, column, value


def load_data_separated() -> Tuple[List[int], List[int], List[int], List[int],
        List[int], List[int]]:
    '''
    Load data from csv.

    :returns: A tuple of lists:
        - list of known rating row indices not used for cross validation
        - list of known rating column indices not used for cross validation
        - list of known ratings not used for cross validation
        - list of known rating row indices used for cross validation
        - list of known rating columns indices used for cross validation
        - list of known ratings used for cross validation
        Note that the lists for known ratings not used for cross validation
        and the lists for known ratings used for cross validation have the
        same format as the input lists.

    '''
    items = []
    users = []
    ratings = []

    # parse data file into three arrays
    with open('../input/data_train.csv', 'rb') as f:
        content = f.readlines()
        content = content[1:]
        for line in content:
            if line:
                row, column, value = parsef(line)
                items.append(column)
                users.append(row)
                ratings.append(value)
    return extract_validation_set(users, items, ratings, 0.1)


def extract_validation_set(users: List[int], items: List[int],
        ratings: List[int], percentage: float) ->  Tuple[List[int],
                List[int], List[int], List[int], List[int], List[int]]:
    '''
    Extract validation set from given ratings, where
    list(zip(users,items,ratings)) results in a list of tuples (r, c, rating),
    where r denotes the row, c denotes the column in the rating matrix A, and
    rating the value at A[r][c].

    :param users: list of known rating row indices
    :param items: list of known rating column indices
    :param ratings: list of known ratings
    :param percentage: percentage of known ratings to use for cross validation
    :returns: A tuple of lists:
        - list of known rating row indices not used for cross validation
        - list of known rating column indices not used for cross validation
        - list of known ratings not used for cross validation
        - list of known rating row indices used for cross validation
        - list of known rating columns indices used for cross validation
        - list of known ratings used for cross validation
        Note that the lists for known ratings not used for cross validation
        and the lists for known ratings used for cross validation have the
        same format as the input lists.
    '''
    users_asked = []
    items_asked = []
    ratings_asked = []

    nr_asked_ratings = int(round(len(ratings) * percentage))

    for i in range(nr_asked_ratings):
        nr_of_ratings = len(ratings)
        index = random.randint(0, nr_of_ratings-1)

        users_asked.append(users[index])
        items_asked.append(items[index])
        ratings_asked.append(ratings[index])

        del users[index]
        del items[index]
        del ratings[index]

    return users, items, ratings, users_asked, items_asked, ratings_asked


def SGD_cross_validate(result: 'numpy.ndarray',
        validation_set: List[Tuple[int, int, int]]) -> float:
    '''
    Compute cross validation score.

    :param result: Rating matrix A
    :param validation_set: Validation set for cross validation. List of tuples
        (r, c, rating), s.t. A[r][c] = rating
    :returns: root mean squared error
    '''
    RMSE = 0.
    for r, c, rating in validation_set:
        RMSE += (result[r-1][c-1] - rating) ** 2
    RMSE /= len(validation_set)
    return RMSE ** 0.5
