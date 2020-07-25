import numpy as np
import random
#from surprise import Dataset
#from surprise import Reader
import pandas as pd
from typing import *

# load the data and insert it into a matrix accordingly
# fill empty entries with the mean of the given data
def load_data(filename):
    nr_of_users = 10000
    nr_of_movies = 1000
    ratings = []
    nr_of_entries = 0
    sum_of_ratings = 0
    with open(filename, 'r') as file:
        file.readline()  # remove the header, don't save it, because we don't need it
        for line in file:
            entry, prediction = line.split(',')  # split the movie rating from the ID
            star_rating = int(prediction)  # save the rating
            row_entry, column_entry = entry.split('_')  # split the ID accordingly (they have the format rX_cY)
            row = int(row_entry[1:])  # remove the 'r'
            column = int(column_entry[1:])  # remove the 'c'
            ratings.append((row-1, column-1, star_rating))  # the IDs are 1-indexed, so subtract 1
            nr_of_entries += 1  # count how many entries there are for the mean calculation
            sum_of_ratings += star_rating  # for calculating the mean
    data_mean = sum_of_ratings / nr_of_entries  # calculate the mean
    data_matrix = np.full((nr_of_users, nr_of_movies), int(data_mean))  # fill every entry with the mean per default
    for (row, column, starRating) in ratings:
        data_matrix[row, column] = starRating  # replace the given entries with the ratings
    return data_matrix, ratings, data_mean


def load_data_zeros(filename):
    nr_of_users = 10000
    nr_of_movies = 1000
    ratings = []
    with open(filename, 'r') as file:
        file.readline()  # remove the header, don't save it, because we don't need it
        for line in file:
            entry, prediction = line.split(',')  # split the movie rating from the ID
            star_rating = int(prediction)  # save the rating
            row_entry, column_entry = entry.split('_')  # split the ID accordingly (they have the format rX_cY)
            row = int(row_entry[1:])  # remove the 'r'
            column = int(column_entry[1:])  # remove the 'c'
            ratings.append((row-1, column-1, star_rating))  # the IDs are 1-indexed, so subtract 1
    data_matrix = np.full((nr_of_users, nr_of_movies), 0.0)  # fill every entry with the mean per default
    for (row, column, starRating) in ratings:
        data_matrix[row, column] = float(starRating)  # replace the given entries with the ratings
    return data_matrix, ratings


# load the data and insert it into a matrix accordingly
# fill empty entries with the mean per user
def load_data_usr_mean(filename):
    nr_of_users = 10000
    nr_of_movies = 1000
    ratings = []
    usr_means = [0] * nr_of_users
    ratings_per_user = [0] * nr_of_users
    with open(filename, 'r') as file:
        file.readline()  # remove the header, don't save it, because we don't need it
        for line in file:
            entry, prediction = line.split(',')  # split the movie rating from the ID
            star_rating = int(prediction)  # save the rating
            row_entry, column_entry = entry.split('_')  # split the ID accordingly (they have the format rX_cY)
            row = int(row_entry[1:])  # remove the 'r'
            column = int(column_entry[1:])  # remove the 'c'
            ratings.append((row-1, column-1, star_rating))  # the IDs are 1-indexed, so subtract 1
            usr_means[row-1] += star_rating
            ratings_per_user[row-1] += 1
    data_matrix = np.full((nr_of_users, nr_of_movies), 0)  # fill every entry with the mean per default
    for i in range(10000):
        usr_means[i] /= ratings_per_user[i]
        for j in range(1000):
            data_matrix[i][j] = usr_means[i]
    for (row, column, starRating) in ratings:
        data_matrix[row][column] = starRating  # replace the given entries with the ratings
    return data_matrix, ratings, usr_means


# load the data and insert it into a matrix accordingly
# fill empty entries with the mean per movie
# each rating is weighted according to the weight array
def load_data_movie_mean(filename, weights):
    nr_of_users = 10000
    nr_of_movies = 1000
    ratings = []
    movie_means = [0] * nr_of_movies
    ratings_per_movie = [0] * nr_of_movies
    with open(filename, 'r') as file:
        file.readline()  # remove the header, don't save it, because we don't need it
        for line in file:
            entry, prediction = line.split(',')  # split the movie rating from the ID
            star_rating = int(prediction)  # save the rating
            row_entry, column_entry = entry.split('_')  # split the ID accordingly (they have the format rX_cY)
            row = int(row_entry[1:])  # remove the 'r'
            column = int(column_entry[1:])  # remove the 'c'
            ratings.append((row-1, column-1, star_rating))  # the IDs are 1-indexed, so subtract 1
            movie_means[column-1] += star_rating * weights[star_rating-1]
            ratings_per_movie[column-1] += weights[star_rating-1]
    data_matrix = np.full((nr_of_users, nr_of_movies), 0)  # fill every entry with the mean per default
    for i in range(1000):
        movie_means[i] /= ratings_per_movie[i]
        for j in range(10000):
            data_matrix[j][i] = movie_means[i]
    for (row, column, starRating) in ratings:
        data_matrix[row][column] = starRating  # replace the given entries with the ratings
    return data_matrix, ratings, movie_means


# return a list of entries that should be predicted
def get_asked_entries():
    asked_entries = []
    with open('../output/sampleSubmission.csv', 'r') as sample_file:
        sample_file.readline()  # throw the header away
        for line in sample_file:
            entry, prediction = line.split(',')
            row_entry, column_entry = entry.split('_')
            row = int(row_entry[1:])
            column = int(column_entry[1:])
            asked_entries.append((row-1, column-1))
    return asked_entries


# store the data according to the sample submission
def store_data(result_matrix):
    file = open('SarahSubmission.csv', 'w+')  # open a new file to write into
    file.write('Id,Prediction\n')  # the header line
    asked_entries = get_asked_entries()
    for (i, j) in asked_entries:
        file.write('r' + str(i+1) + '_c' + str(j+1) + ',' + str(int(round(result_matrix[i][j]))) + '\n')  # store with the same ID as the sample


# store the data according to the sample submission
def store_data_float(result_matrix, file_name: str):
    file = open(file_name, 'w+')  # open a new file to write into
    file.write('Id,Prediction\n')  # the header line
    asked_entries = get_asked_entries()
    for (i, j) in asked_entries:
        file.write('r' + str(i+1) + '_c' + str(j+1) + ',' + str(result_matrix[i][j]) + '\n')  # store with the same ID as the sample


# randomly pick 10% of the entries for cross validation
def split_data(data_matrix, ratings, data_mean):
    nr_of_ratings = len(ratings)
    nr_of_asked_entries = int(round(0.1 * nr_of_ratings))  # use 10% of the given entries for testing
    asked_entries = random.sample(ratings, nr_of_asked_entries)  # randomly sample the 10% we want to fill in
    pred_matrix = data_matrix
    for (row, column, star_rating) in asked_entries:
        pred_matrix[row][column] = data_mean[column]
    return pred_matrix, asked_entries

# randomly pick 10% of the users for cross validation
def split_users(data_matrix, data_mean):
    usrs = [i for i in range(10000)]
    asked_users = random.sample(usrs, 1000)
    pred_matrix = data_matrix
    for i in asked_users:
        for j in range(1000):
            pred_matrix[i][j] = data_mean
    return pred_matrix, asked_users


def shrink(nr_of_ratings, means, ratings):
    alpha = 1

    mean = 0

    for (row, column, star_rating) in ratings:
        mean += star_rating

    mean /= len(ratings)

    for i in range(1000):
        means[i] = (alpha/(alpha + nr_of_ratings[i])) * mean + (nr_of_ratings[i] / (alpha + nr_of_ratings[i])) * means[i]

    return means


# center the data by removing a bias term
# method used: global mean rating
def center_global_mean(data, ratings):
    mean = 0

    for (row, column, star_rating) in ratings:
        mean += star_rating

    mean /= len(ratings)

    for (row, column, star_rating) in ratings:
        data[row, column] -= mean

    return data


# center the data by removing a bias term
# method used: movie mean rating
def center_movie_mean(data, ratings):
    movie_mean = [0 for i in range(1000)]
    movie_ratings = [0 for i in range(1000)]

    for (row, column, star_rating) in ratings:
        movie_mean[column] += star_rating
        movie_ratings[column] += 1

    for i in range(1000):
        movie_mean[i] /= movie_ratings[i]

    for (row, column, star_rating) in ratings:
        data[row, column] -= movie_mean[column]

    return data


# center the data by removing a bias term
# method used: user mean rating
def center_user_mean(data, ratings):
    user_mean = [0 for i in range(10000)]
    user_ratings = [0 for i in range(10000)]

    for (row, column, star_rating) in ratings:
        user_mean[row] += star_rating
        user_ratings[row] += 1

    for i in range(10000):
        user_mean[i] /= user_ratings[i]

    for (row, column, star_rating) in ratings:
        data[row, column] -= user_mean[row]

    return data


def compute_deviation(ratings):
    movie_mean = [0 for i in range(1000)]
    movie_ratings = [0 for i in range(1000)]

    for (row, column, star_rating) in ratings:
        movie_mean[column] += star_rating
        movie_ratings[column] += 1

    for i in range(1000):
        movie_mean[i] /= movie_ratings[i]

    movie_mean = shrink(movie_ratings, movie_mean, ratings)

    user_mean = [0 for i in range(10000)]
    user_ratings = [0 for i in range(10000)]

    for (row, column, star_rating) in ratings:
        user_mean[row] += (star_rating - movie_mean[column])
        user_ratings[row] += 1

    for i in range(10000):
        user_mean[i] /= user_ratings[i]

    user_mean = shrink(user_ratings, user_mean, ratings)

    return user_mean, movie_mean


# center the data by removing a bias term
# method used: movie mean and user mean deviation from movie mean
def center_deviation_movie_mean(data, ratings):
    user_mean, movie_mean = compute_deviation(ratings)

    for (row, column, star_rating) in ratings:
        data[row, column] -= (user_mean[row] + movie_mean[column])

    return data


def reverse_centering_deviation(data, ratings):
    user_mean, movie_mean = compute_deviation(ratings)

    for i in range(10000):
        for j in range(1000):
            data[i, j] += (user_mean[i] + movie_mean[j])

    return data


def parsef(line):
    l1 = line.decode('utf8').split(',')
    l2 = l1[0].split('_')
    row = int(l2[0][1:])
    column = int(l2[1][1:])
    value = float(l1[1])
    return row, column, value


def load_data_raw():
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


def load_data_surprise():
    items, users, ratings = load_data_raw()

    # Creation of the dataframe. Column names are irrelevant.
    ratings_dict = {'itemID': items,
                    'userID': users,
                    'rating': ratings}
    df = pd.DataFrame(ratings_dict)

    # The columns must correspond to user id, item id and ratings (in that order).
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader=reader)
    return data
