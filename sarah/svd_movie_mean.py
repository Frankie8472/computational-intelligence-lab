# movie rating prediction based on given matrix entries
# 10'000 Users (rows)
# 1'000 Movies (columns)

import matplotlib.pyplot as plt
from sarah.inputhandler import *

nr_of_users = 10000
nr_of_movies = 1000


# the prediction for any missing value X[i,j] can be computed as the inner product of the ith row in U and the jth
# column in V
def predict_values(asked_entries, u, v, matrix):
    for (i, j) in asked_entries:
        matrix[i][j] = int(round(u[i, :].dot(v[:, j])))
    return matrix


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


weights = [1, 1, 1, 1, 1]
loaded_matrix, given_ratings, mean = load_data_movie_mean('../input/data_train.csv', weights)
U, V = compute_svd(loaded_matrix, 8)
asked_entries = get_asked_entries()
result_matrix = predict_values(asked_entries, U, V, loaded_matrix)
store_data(result_matrix)
