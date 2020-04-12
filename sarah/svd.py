########################################################################################################################
# This is the first improvement using SVD from the exercise sheet 2                                                    #
# Searches through every possible value for truncating the diagonal SVD-matrix and looks for the one with the          #
# smallest RMSE-Error                                                                                                  #
# For this purpose, the known values are splitted 90-10 for cross-validation                                           #
########################################################################################################################

# movie rating prediction based on given matrix entries
# 10'000 Users (rows)
# 1'000 Movies (columns)

import numpy as np
import matplotlib.pyplot as plt
import random

nr_of_users = 10000
nr_of_movies = 1000


def prepare_training(ratings, data_mean, data_matrix):
    nr_of_ratings = len(ratings)
    nr_of_asked_entries = int(round(0.1 * nr_of_ratings))   # use 10% of the given entries for testing
    asked_entries = random.sample(ratings, nr_of_asked_entries)  # randomly sample the 10% we want to fill in
    pred_matrix = data_matrix
    for (row, column, star_rating) in asked_entries:
        pred_matrix[row][column] = data_mean
    return pred_matrix, asked_entries


def load_data(filename):
    nr_of_entries = 0
    sum_of_ratings = 0
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
            nr_of_entries += 1  # count how many entries there are for the mean calculation
            sum_of_ratings += star_rating  # for calculating the mean
    data_mean = sum_of_ratings / nr_of_entries  # calculate the mean
    # print('the data mean is: ' + str(data_mean))
    data_matrix = np.full((nr_of_users, nr_of_movies), int(data_mean))  # fill every entry with the mean per default
    for (row, column, starRating) in ratings:
        data_matrix[row, column] = starRating  # replace the given entries with the ratings
    return data_matrix, ratings, data_mean


# the prediction for any missing value X[i,j] can be computed as the inner product of the ith row in U and the jth
# column in V
def predict_value(row_nr, col_nr, u, v):
    return int(round(u[row_nr, :].dot(v[:, col_nr])))


def store_data(u, v):
    file = open('SarahSubmission.csv', 'w+')  # open a new file to write into
    file.write('Id,Prediction\n')  # the header line
    with open('sampleSubmission.csv', 'r') as sample_file:  # open the sample file to see which values I have to submit
        sample_file.readline()  # throw the header away
        for line in sample_file:
            entry, prediction = line.split(',')
            row_entry, column_entry = entry.split('_')
            row = int(row_entry[1:]) - 1
            column = int(column_entry[1:]) - 1
            file.write(entry + ',' + str(predict_value(row, column, u, v)) + '\n')  # store with same ID as the sample


def truncate(s, to_truncate):
    n = s.shape[0]
    # print(s.shape)
    return np.append(s[:to_truncate], np.zeros(n-to_truncate))


def plot(values):
    y = []
    for i in range(len(values)):
        if values[i] > 1:
            y.append(values[i])
    x = [i for i in range(len(y))]
    plt.plot(x, y)
    plt.show()
    print(values)


def compute_svd(data_matrix, to_truncate):
    u, s, v_t = np.linalg.svd(data_matrix, full_matrices=True)  # compute the svd
    # plot(s[1:])
    s = truncate(s, to_truncate)  # specify how many singular values you want to keep
    s_filled = np.zeros((u.shape[0], v_t.shape[0]))  # create a matrix for the eigenvalues
    s_filled[:min(u.shape[0], v_t.shape[0]), :min(u.shape[0], v_t.shape[0])] = np.diag(s)  # fill sigma with EV
    # print(u.shape, s.shape, s_filled.shape, v_t.shape)
    u_ = np.dot(u, s_filled)  # multiply with singular values matrix
    return u_, v_t


def fill_matrix(pred_matrix, asked, relevant):
    summed = 0
    u, v = compute_svd(pred_matrix, relevant)
    for (row, column, star_rating) in asked:
        summed += ((predict_value(row, column, u, v) - star_rating) ** 2)
    return (summed / len(asked)) ** 0.5


loaded_matrix, given_ratings, mean = load_data('data_train.csv')
prediction_matrix, to_fill = prepare_training(given_ratings, mean, loaded_matrix)
minimum = current = 100000.0
min_k = 10000000
for k in range(1000):
    current = fill_matrix(prediction_matrix, to_fill, k)
    if current < minimum:
        minimum = current
        min_k = k
    print(str(k) + ": " + str(current))
print(str(min_k) + ": " + str(minimum))
# U, V = compute_svd(loaded_matrix)
# store_data(U, V)
