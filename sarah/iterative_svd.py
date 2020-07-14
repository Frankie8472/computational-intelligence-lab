# movie rating prediction based on given matrix entries
# 10'000 Users (rows)
# 1'000 Movies (columns)

import matplotlib.pyplot as plt
from sarah.inputhandler import *

nr_of_users = 10000
nr_of_movies = 1000


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
    # print(values)


def compute_svd(data_matrix):
    u, s, v_t = np.linalg.svd(data_matrix, full_matrices=True)  # compute the svd
    plot(s[1:])
    to_truncate = int(input("how many singular values should we keep: "))
    s = truncate(s, to_truncate)  # specify how many singular values you want to keep
    s_filled = np.zeros((u.shape[0], v_t.shape[0]))  # create a matrix for the eigenvalues
    s_filled[:min(u.shape[0], v_t.shape[0]), :min(u.shape[0], v_t.shape[0])] = np.diag(s)  # fill sigma with EV
    # print(u.shape, s.shape, s_filled.shape, v_t.shape)
    u_ = np.dot(u, s_filled)  # multiply with singular values matrix
    return u_, v_t


weights = [100, 10, 1, 10, 100]
loaded_matrix, given_ratings, movie_means = load_data_movie_mean('../input/data_train.csv', weights)

for i in range(10):
    # compute the SVD as in the baseline
    U, V = compute_svd(loaded_matrix)
    # compute the resulting matrix
    loaded_matrix = np.dot(U, V)
    # determine the new movie means
    movie_means = [0] * 1000
    for j in range(1000):
        for k in range(10000):
            movie_means[j] += loaded_matrix[k][j]
        movie_means[j] /= 10000
    # make a new matrix out of the movie means
    for j in range(1000):
        for k in range(10000):
            loaded_matrix[k][j] = movie_means[j]
    # reset the given matrices to the input value
    for (row, column, starRating) in given_ratings:
        loaded_matrix[row][column] = starRating

store_data(loaded_matrix)
