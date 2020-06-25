import numpy as np
import random


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

