import numpy as np
import random
from typing import *

def get_asked_entries() -> List[Tuple[int, int]]:
    """
    Return a list of entries that should be predicted.

    :returns: List of tuples (r, c) for each asked entry, where r is the row
        index and c i the column index of the asked entry
    """
    asked_entries = []
    with open('../../input/sampleSubmission.csv', 'r') as sample_file:
        sample_file.readline()  # throw the header away
        for line in sample_file:
            entry, prediction = line.split(',')
            row_entry, column_entry = entry.split('_')
            row = int(row_entry[1:])
            column = int(column_entry[1:])
            asked_entries.append((row-1, column-1))
    return asked_entries


def store_data_float(result_matrix: 'numpy.ndarray', file_name: str) -> None:
    """
    Write submission file.

    :param result_matrix: prediction matrix
    :param file_name: path to output file
    """
    file = open(file_name, 'w+')  # open a new file to write into
    file.write('Id,Prediction\n')  # the header line
    asked_entries = get_asked_entries()
    for (i, j) in asked_entries:
        file.write('r' + str(i+1) + '_c' + str(j+1) + ',' +
                str(result_matrix[i][j]) + '\n')  # store with the same ID as the sample


def load_data_movie_mean(filename: str, weights: List[int]) -> (np.ndarray,
        List[Tuple[int, int, int]], List[int]):
    '''
        Load rating data into a matrix.

        :param filename: path to data
        :param weights: list of weights [w1, w2, w3, w4, w5], where wi is
            the weight for rating value i
        :returns:
            data_matrix: matrix containing known ratings
            ratings: list of tuples (row, column, rating) of known ratings
            movie_means: list of weighted movie means where mean[i] is the
                mean of movie i
        '''
    nr_of_users = 10000
    nr_of_movies = 1000
    ratings = []
    movie_means = [0] * nr_of_movies
    ratings_per_movie = [0] * nr_of_movies

    with open(filename, 'r') as file:
        file.readline()  # discard header
        for line in file:
            entry, prediction = line.split(',')
            star_rating = int(prediction)
            row_entry, column_entry = entry.split('_')
            row = int(row_entry[1:])
            column = int(column_entry[1:])
            ratings.append((row-1, column-1, star_rating)) # IDs are 1-indexed
            movie_means[column-1] += star_rating * weights[star_rating-1]
            ratings_per_movie[column-1] += weights[star_rating-1]
    # Fill matrix with movie means 
    data_matrix = np.full((nr_of_users, nr_of_movies), 0)
    for i in range(1000):
        movie_means[i] /= ratings_per_movie[i]
        for j in range(10000):
            data_matrix[j][i] = movie_means[i]
    # Replace movie mean with the rating if a rating is given by the data
    # set
    for (row, column, starRating) in ratings:
        data_matrix[row][column] = starRating

    return data_matrix, ratings, movie_means


def center_deviation_movie_mean(data: np.ndarray,
        ratings: List[Tuple[int, int, int]]) -> np.ndarray:
    '''
    Center data by removing a bias term (movie and user mean deviation from
    movie mean)

    :param data: rating matrix
    :param ratings: list of rating tuples (row, column, rating) of known
        ratings
    :retuns: centered rating matrix
    '''
    user_mean, movie_mean = compute_deviation(ratings)

    for (row, column, star_rating) in ratings:
        data[row, column] -= (user_mean[row] + movie_mean[column])

    return data


def reverse_centering_deviation(data: np.ndarray,
        ratings: List[Tuple[int, int, int]]) -> np.ndarray:
    '''
    Reverse centering by removing a bias term (movie and user mean
    deviation from movie mean)

    :param data: centered rating matrix
    :param ratings: list of tuples (row, column, ratings) of known ratings
    :returns: decentered rating matrix
    '''
    user_mean, movie_mean = compute_deviation(ratings)

    for i in range(10000):
        for j in range(1000):
            data[i, j] += (user_mean[i] + movie_mean[j])

    return data


def compute_deviation(ratings: List[Tuple[int, int, int]]) -> (List[int],
        List[int]):
    '''
        Compute user and movie mean deviations from movie mean.

        :param ratings: list of tuples (row, column, rating) of known
            ratings
        :returns:
            user_mean: list of user means, where user_mean[i] is the mean
                for user i
            movie_mean: list of movie means, where movie_mean[i] is the
                mean for movie i
    '''
    # Compute movie means
    movie_mean = [0 for i in range(1000)]
    movie_ratings = [0 for i in range(1000)]

    for (row, column, star_rating) in ratings:
        movie_mean[column] += star_rating
        movie_ratings[column] += 1

    for i in range(1000):
        movie_mean[i] /= movie_ratings[i]

    movie_mean = shrink(movie_ratings, movie_mean, ratings)

    # Compute user means
    user_mean = [0 for i in range(10000)]
    user_ratings = [0 for i in range(10000)]

    for (row, column, star_rating) in ratings:
        user_mean[row] += (star_rating - movie_mean[column])
        user_ratings[row] += 1

    for i in range(10000):
        user_mean[i] /= user_ratings[i]

    user_mean = shrink(user_ratings, user_mean, ratings)

    return user_mean, movie_mean


def shrink(nr_of_ratings: List[int], means: List[float],
        ratings: List[Tuple[int, int, int]]) -> List[float]:
    '''
    Apply shrinkage.

    :param nr_of_ratings: list where nr_of_ratings[i] is the number of
        known ratings of i
    :param means: list where means[i] is the mean of i
    :param ratings: list of tuples (row, column, rating) of known ratings
    :returns: list of means, where means[i] is the mean of i after
        shrinkage
    '''
    alpha = 1
    mean = 0

    for (row, column, star_rating) in ratings:
        mean += star_rating

    mean /= len(ratings)

    for i in range(1000):
        means[i] = ((alpha/(alpha + nr_of_ratings[i])) * mean +
                (nr_of_ratings[i] / (alpha + nr_of_ratings[i])) * means[i])

    return means
