import numpy as np
import pandas as pd

import data_util as util

'''
STEP 1: What value(s) should we use for unknown ranking?
Using cross validation, we compare the score of different approaches.
Save matrix with filled in values as csv.
'''

def fill(data_path: str, users: int, movies: int, output_path: str,
        cross_validate: bool = False) -> None:
    '''
    This function is called by run.py to fill unknown ratings with values.
    Writes the full matrix to a csv in the same format as data_path to
    output_path.

    :param data_path: path to rating csv
    :param users: number of users
    :param movies: number of movies
    :param output_path: path to csv to which the matrix should be written to
    :param cross_validate: True, if cross validation should be done. No output
        file will then be written.
    '''
    if cross_validate:
        print('Matrix filling approach:')
    A, ratings = util.csv_to_rating_matrix(data_path, np.nan, users, movies)
    approach_id = 5
    A = do_approach[approach_id](A, ratings, users, movies, cross_validate,
            data_path)
    if not cross_validate:
        util.create_submission_csv(A, output_path)



def same_value(A: 'numpy.ndarray', ratings: set, users: int, movies: int,
        cross_validate: bool, data_path: str = None) -> 'np.ndarray':
    '''
    Use same value for all unknowns.

    :param A: rating matrix with np.nan as unknown ratings
    :param ratings: set of known rating tuples (row, column, rating)
    :param user: number of users
    :param movies: number of movies
    :param cross_validate: True, if cross validation should be done. No output
        file will then be written.
    :param data_path: path to input rating csv
    :return: full matrix
    '''
    unknown = 5
    np.nan_to_num(A, copy=False, nan=unknown)
    if cross_validate:
        A_split, asked_entries = util.split_data(A, ratings, np.full((movies,
            1), unknown), True)
        score = util.score(A_split, asked_entries)
        print('Same value\nScore: '+str(score))
    return A


def mean_of_all(A: 'numpy.ndarray', ratings: set, users: int, movies: int,
        cross_validate: bool, data_path: str) -> 'np.ndarray':
    '''
    Use mean of all known values.

    :param A: rating matrix with np.nan as unknown ratings
    :param ratings: set of known rating tuples (row, column, rating)
    :param user: number of users
    :param movies: number of movies
    :param cross_validate: True, if cross validation should be done. No output
        file will then be written.
    :param data_path: path to input rating csv
    :return: full matrix
    '''
    mean = 0
    data = pd.read_csv(data_path, header=0)
    mean = data.sum(numeric_only=True)
    mean = int(round(mean.iloc[0]/data.shape[0]))
    A, ratings = util.csv_to_rating_matrix(data_path, mean, users, movies)
    if cross_validate:
        A_split, asked_entries = util.split_data(A, ratings, np.full((movies,
            1), mean), True)
        score = util.score(A_split, asked_entries)
        print('Mean of all ratings\nScore: '+str(score))
    return A


def mean_of_user(A: 'numpy.ndarray', ratings: set, users: int, movies: int,
        cross_validate: bool, data_path: str) -> 'np.ndarray':
    '''
    Use mean of each user.

    :param A: rating matrix with np.nan as unknown ratings
    :param ratings: set of known rating tuples (row, column, rating)
    :param user: number of users
    :param movies: number of movies
    :param cross_validate: True, if cross validation should be done. No output
        file will then be written.
    :param data_path: path to input rating csv
    :return: full matrix
    '''
    mean = np.zeros((users,1))
    for i in range(users):
        mean[i] = int(round(np.nanmean(A[i,:])))
    np.nan_to_num(A, copy=False, nan=0)
    for i in range(users):
        np.putmask(A[i,:], A[i,:] == 0, mean[i])
    if cross_validate:
        A_split, asked_entries = util.split_data(A, ratings, mean, False)
        score = util.score(A_split, asked_entries)
        print('Mean of each user\nScore: '+str(score))
    return A


def mean_of_movie(A: 'numpy.ndarray', ratings: set, users: int, movies: int,
        cross_validate: bool, data_path: str) -> 'np.ndarray':
    '''
    Use mean of each movie.

    :param A: rating matrix with np.nan as unknown ratings
    :param ratings: set of known rating tuples (row, column, rating)
    :param user: number of users
    :param movies: number of movies
    :param cross_validate: True, if cross validation should be done. No output
        file will then be written.
    :param data_path: path to input rating csv
    :return: full matrix
    '''
    mean = np.zeros((movies,1))
    for i in range(movies):
        mean[i] = int(round(np.nanmean(A[:,i])))
    np.nan_to_num(A, copy=False, nan=0)
    for i in range(movies):
        np.putmask(A[:,i], A[:,i]==0, mean[i])
    if cross_validate:
        A_split, asked_entries = util.split_data(A, ratings, mean, True)
        score = util.score(A_split, asked_entries)
        print('Mean of each movie\nScore: '+str(score))
    return A


def mean_of_movie_adjusted(A: 'numpy.ndarray', ratings: set, users: int,
        movies: int,
        cross_validate: bool, data_path: str) -> 'np.ndarray':
    '''
    Use mean of each movie. Increase all ratings of a user if they rated enough
    movies (above a certain threshold) and their average is >=4. Decrease if 
    average is <= 2. Reason: Some users may tend to give low or high ratings.

    :param A: rating matrix with np.nan as unknown ratings
    :param ratings: set of known rating tuples (row, column, rating)
    :param user: number of users
    :param movies: number of movies
    :param cross_validate: True, if cross validation should be done. No output
        file will then be written.
    :param data_path: path to input rating csv
    :return: full matrix
    '''
    A_original = A.copy()   # matrix with NaN as unknowns
    # Compute movie and user means
    movie_mean = np.zeros((movies,1))
    for i in range(movies):
        movie_mean[i] = int(round(np.nanmean(A[:,i])))
    user_mean = np.zeros((users,1))
    for i in range(users):
        user_mean[i] = int(round(np.nanmean(A[i,:])))
    # Count number of ratings per user
    np.nan_to_num(A, copy=False, nan=0)
    user_ratings = np.zeros(users)
    for i in range(users):
        user_ratings[i] = np.count_nonzero(A[i,:])
    # Plug in movie mean into unknowns
    for i in range(movies):
        np.putmask(A[:,i], A[:,i]==0, movie_mean[i])
    # Adjust rating
    threshold = movies/2
    for i in range(users): 
        if user_ratings[i] > threshold:
            # Increase
            if user_mean[i] >= 4:
                for j in range(movies):
                    if np.isnan(A_original[i,j]):
                           A[i,j] = min(A[i,j]+1, 5)
            # Decrease
            if user_mean[i] <= 2:
                for j in range(movies):
                    if np.isnan(A_original[i,j]):
                        A[i,j] = max(A[i,j]-1, 1)
    if cross_validate:
        A_split, asked_entries = util.split_data(A, ratings, movie_mean, True)
        score = util.score(A_split, asked_entries)
        print('Adjusted mean of each movie\nScore: '+str(score))
    return A


do_approach = { 1: same_value,              # Score: 1.13
                2: mean_of_all,             # Score: 1.13
                3: mean_of_user,            # Score: 1.11
                4: mean_of_movie,           # Score: 1.07
                5: mean_of_movie_adjusted   # Score: 1.07
                }
