'''
Utility functions for handling data
'''
import numpy as np
import pandas as pd
import re
import random
from typing import *

def csv_to_rating_matrix(data_path: str, unknown: int, users: int,
        movies: int) -> ('numpy.ndarray', Set[Tuple[int, int, int]]): 
    '''
    Create matrix out of csv file, fill in unknown ratings.

    :param data_path: path to csv
    :param unknown: value for unknown ratings
    :param users: number of users
    :param movies: number of movies
    :return: rating matrix, set of rating tuples (row, column, rating)
    '''
    data = pd.read_csv(data_path, header=0)

    A = np.full((users,movies), unknown)
    ratings = set()
    for r in range(data.shape[0]):
        position = data.iloc[r,0]
        i = int(re.search('r[0-9]*_',position).group(0)[1:-1])-1
        j = int(re.search('c[0-9]*',position).group(0)[1:])-1
        A[i,j] = data.iloc[r,1]
        ratings.add((i, j , A[i,j]))
    return A, ratings


def create_submission_csv(A: 'numpy.ndarray', output_path: str) -> None:
    '''
    Create submission file.

    :param A: rating matrix
    :param output_path: path to output csv
    '''
    unknowns = pd.read_csv('../../input/sampleSubmission.csv')
    file = open(output_path, 'w+')
    file.write('Id,Prediction\n')
    for u in range(unknowns.shape[0]):
        position = unknowns.iloc[u,0]
        i = int(re.search('r[0-9]*_',position).group(0)[1:-1])-1
        j = int(re.search('c[0-9]*',position).group(0)[1:])-1
        file.write(position+','+str(A[i,j])+'\n')
    return


def split_data(data_matrix: 'numpy.ndarray',
        ratings: Set[Tuple[int, int ,int]], data_mean: 'numpdy.ndarray'
        ) -> ('numpy.ndarray', List[Tuple[int, int, int]]):
    '''
    Split up data for cross validation. 10% is used as validation set.
    
    :param data_matrix: rating matrix
    :param set of known rating tuples (row, column, rating)
    :param data_mean: matrix of size (movies, 1) containing movie rating means
        of known ratings 
    :returns:
        pred_matrix: prediction matrix where ratings used for cross validation
            are set to the mean
        asked_entries: validation set for cross validation, list of tuples
            (row, column, rating)
    '''
    nr_of_ratings = len(ratings)
    nr_of_asked_entries = int(round(0.1 * nr_of_ratings))
    asked_entries = random.sample(ratings, nr_of_asked_entries)
    pred_matrix = data_matrix.copy()
    for (row, column, star_rating) in asked_entries:
            pred_matrix[row][column] = data_mean[column]
    return pred_matrix, asked_entries


def score(result: 'numpy.ndarray', asked: Set[Tuple[int, int, int]]) -> float:
    '''
        Compute cross validation score.

        :param result: prediction matrix
        :param asked: validation set, set of tuples (row, column, rating)
        :returns: score
    '''
    RMSE = 0.
    for (r, c, rating) in asked:
        assert not np.isnan(result[r][c])
        RMSE += (result[r][c] - rating) ** 2 
    RMSE /= len(asked)
    return RMSE ** 0.5


