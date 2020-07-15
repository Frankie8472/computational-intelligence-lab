import numpy as np
import pandas as pd
import random

import data_util as util

'''
STEP 1: What value(s) should we use for unknown ranking?
Using cross validation, we compare the score of different approaches.
'''

def fill(A: 'numpy.ndarray', ratings: set, data_path: str, users: int,
        movies: int) -> 'numpy.ndarray':
    '''
    This function is called by run.py to fill unknown ratings with values.

    :param A: input rating matrix with unknown ratings set to NaN
    :param ratings: set of known ratings tuples (row, column, rating)
    :param data_path: path to rating csv
    :param users: number of users
    :param movies: number of movies
    :return: full rating matrix
    '''
    approach_id = 5
    A = do_approach[approach_id](A, ratings, users, movies, data_path)
    return A


def same_value(A: 'numpy.ndarray', ratings: set, users: int, movies: int,
        data_path: str) -> 'numpy.ndarray':
    '''
    Use same value for all unknowns.

    :param A: rating matrix with np.nan as unknown ratings
    :param ratings: set of known rating tuples (row, column, rating)
    :param user: number of users
    :param movies: number of movies
    :param data_path: path to input rating csv
    :return: full rating matrix
    '''
    print('Matrix filling approach: same value')
    unknown = 4
    np.nan_to_num(A, copy=False, nan=unknown)
    return A


def mean_of_all(A: 'numpy.ndarray', ratings: set, users: int, movies: int,
        data_path: str) -> 'numpy.ndarray':
    '''
    Use mean of all known values.

    :param A: rating matrix with np.nan as unknown ratings
    :param ratings: set of known rating tuples (row, column, rating)
    :param user: number of users
    :param movies: number of movies
    :param data_path: path to input rating csv
    :return: full rating matrix
    '''
    print('Matrix filling approach: Mean of all ratings')
    mean = 0
    data = pd.read_csv(data_path, header=0)
    mean = data.sum(numeric_only=True)
    mean = mean.iloc[0]/data.shape[0]
    np.nan_to_num(A, copy=False, nan=mean)
    return A


def mean_of_user(A: 'numpy.ndarray', ratings: set, users: int, movies: int,
        data_path: str) -> 'numpy.ndarray':
    '''
    Use mean of each user.

    :param A: rating matrix with np.nan as unknown ratings
    :param ratings: set of known rating tuples (row, column, rating)
    :param user: number of users
    :param movies: number of movies
    :param data_path: path to input rating csv
    :return: full rating matrix
    '''
    print('Matrix filling approach: Mean of each user')
    mean = np.zeros((users,1))
    for i in range(users):
        mean[i] = int(round(np.nanmean(A[i,:])))
    np.nan_to_num(A, copy=False, nan=0)
    for i in range(users):
        np.putmask(A[i,:], A[i,:] == 0, mean[i])
    return A


def mean_of_movie(A: 'numpy.ndarray', ratings: set, users: int, movies: int,
       data_path: str) -> 'numpy.ndarray':
    '''
    Use mean of each movie.

    :param A: rating matrix with np.nan as unknown ratings
    :param ratings: set of known rating tuples (row, column, rating)
    :param user: number of users
    :param movies: number of movies
    :param data_path: path to input rating csv
    :return: full rating matrix
    '''
    print('Matrix filling approach: Mean of each movie')
    mean = np.zeros((movies,1))
    for i in range(movies):
        mean[i] = np.nanmean(A[:,i])
    np.nan_to_num(A, copy=False, nan=0)
    for i in range(movies):
        np.putmask(A[:,i], A[:,i]==0, mean[i])
    return A


def mean_of_movie_adjusted(A: 'numpy.ndarray', ratings: set, users: int,
        movies: int, data_path: str) -> 'numpy.ndarray':
    '''
    Use mean of each movie. Increase all ratings of a user if they rated enough
    movies (above a certain threshold) and their average is >=4. Decrease if 
    average is <= 2. Reason: Some users may tend to give low or high ratings.

    :param A: rating matrix with np.nan as unknown ratings
    :param ratings: set of known rating tuples (row, column, rating)
    :param user: number of users
    :param movies: number of movies
    :param data_path: path to input rating csv
    :return: full rating matrix
    '''
    print('Matrix filling approach: mean of each movie, adjusted per user')
    A_original = A.copy()
    # Compute movie and user means
    movie_mean = np.zeros((movies,1))
    for i in range(movies):
        movie_mean[i] = np.nanmean(A[:,i])
    user_mean = np.zeros((users,1))
    for i in range(users):
        user_mean[i] = np.nanmean(A[i,:])
    # Count number of ratings per user
    np.nan_to_num(A, copy=False, nan=0)
    user_ratings = np.zeros(users)
    for i in range(users):
        user_ratings[i] = np.count_nonzero(A[i,:])
    # Plug in movie mean into unknowns
    for i in range(movies):
        np.putmask(A[:,i], A[:,i]==0, movie_mean[i])
    # Adjust rating
    threshold = movies/3
    for i in range(users): 
        if user_ratings[i] > threshold:
            # Increase
            if user_mean[i] > 4 :
                for j in range(movies):
                    if np.isnan(A_original[i,j]):
                        orig = A[i,j]
                        A[i,j] = min(A[i,j]+1, 5)
            # Decrease
            if user_mean[i] < 2 :
                for j in range(movies):
                    if np.isnan(A_original[i,j]):
                        A[i,j] = max(A[i,j]-1, 1)
    return A


do_approach = { 1: same_value,              # Score: 1.13
                2: mean_of_all,             # Score: 1.12
                3: mean_of_user,            # Score: 1.12
                4: mean_of_movie,           # Score: 1.03
                5: mean_of_movie_adjusted   # Score: 1.03
                }
