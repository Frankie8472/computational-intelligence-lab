'''
Utility functions for handling data
'''
import numpy as np
import pandas as pd
import re
import random

def csv_to_rating_matrix(data_path: str, unknown: int, users: int, movies: int) -> ('numpy.ndarray', set): 
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
    Create submission file.

    :param A: rating matrix
    :param id: submission ID
    '''
    unknowns = pd.read_csv('../input_data/sampleSubmission.csv')
    file = open(output_path, 'w+')
    file.write('Id,Prediction\n')
    for u in range(unknowns.shape[0]):
        position = unknowns.iloc[u,0]
        i = int(re.search('r[0-9]*_',position).group(0)[1:-1])-1
        j = int(re.search('c[0-9]*',position).group(0)[1:])-1
        file.write(position+','+str(A[i,j])+'\n')
    return


# --- EXCERPT FROM SARAH'S GENIUS BRAIN ---

# randomly pick 10% of the entries for cross validation
def split_data(data_matrix, ratings, data_mean, c):
    nr_of_ratings = len(ratings)
    nr_of_asked_entries = int(round(0.1 * nr_of_ratings))  # use 10% of the given entries for testing
    asked_entries = random.sample(ratings, nr_of_asked_entries)  # randomly sample the 10% we want to fill in
    pred_matrix = data_matrix.copy()
    for (row, column, star_rating) in asked_entries:
        if c:
            pred_matrix[row][column] = data_mean[column]
        else:
            pred_matrix[row][column] = data_mean[row]
    return pred_matrix, asked_entries

def score(result, asked):
    RMSE = 0.
    for (r, c, rating) in asked:
        RMSE += (result[r][c] - rating) ** 2
    RMSE /= len(asked)
    return RMSE ** 0.5


