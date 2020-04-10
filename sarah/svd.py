########################################################################################################################
# This is the first improvement using SVD from the exercise sheet 2                                                    #
########################################################################################################################

# movie rating prediction based on given matrix entries
# 10'000 Users (rows)
# 1'000 Movies (columns)

import numpy as np

nr_of_users = 10000
nr_of_movies = 1000

ratings = []


def load_data(filename):
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
    print('the data mean is: ' + str(data_mean))
    data_matrix = np.full((nr_of_users, nr_of_movies), int(data_mean))  # fill every entry with the mean per default
    for (row, column, starRating) in ratings:
        data_matrix[row, column] = starRating  # replace the given entries with the ratings
    return data_matrix


# the prediction for any missing value X[i,j] can be computed as the inner product of the ith row in U and the jth
# column in V
def predict_value(row_nr, col_nr, u, v):
    return int(round(u[row_nr, :].dot(v[:, col_nr])))


def store_data(U, V):
    file = open('SarahSubmission.csv', 'w+')  # open a new file to write into
    file.write('Id,Prediction\n')  # the header line
    with open('sampleSubmission.csv', 'r') as sample_file:  # open the sample file to see which values I have to submit
        sample_file.readline()  # throw the header away
        for line in sample_file:
            entry, prediction = line.split(',')
            row_entry, column_entry = entry.split('_')
            row = int(row_entry[1:]) - 1
            column = int(column_entry[1:]) - 1
            file.write(entry + ',' + str(predict_value(row, column, U, V)) + '\n')  # store with the same ID as the sample


def truncate(s, k):
    n = s.shape[0]
    print(s.shape)
    return np.append(s[:k], np.zeros(n-k))


def compute_svd(data_matrix):
    u, s, v_t = np.linalg.svd(data_matrix, full_matrices=True)  # compute the svd
    s = truncate(s, 5)  # specify how many singular values you want to keep
    s_filled = np.zeros((u.shape[0], v_t.shape[0]))  # create a matrix for the eigenvalues
    s_filled[:min(u.shape[0], v_t.shape[0]), :min(u.shape[0], v_t.shape[0])] = np.diag(s)  # fill sigma with EV
    print(u.shape, s.shape, s_filled.shape, v_t.shape)
    u_ = np.dot(u, s_filled)  # multiply with singular values matrix
    return u_, v_t


loaded_matrix = load_data('data_train.csv')
U, V = compute_svd(loaded_matrix)
store_data(U, V)
