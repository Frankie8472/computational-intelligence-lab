# movie rating prediction based on given matrix entries
# 10'000 Users (rows)
# 1'000 Movies (columns)

# best "to_truncate" value so far: keep the first 41 singular values

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
    # to_truncate = int(input("how many singular values should we keep: "))
    s = truncate(s, to_truncate)  # specify how many singular values you want to keep
    s_filled = np.zeros((u.shape[0], v_t.shape[0]))  # create a matrix for the eigenvalues
    s_filled[:min(u.shape[0], v_t.shape[0]), :min(u.shape[0], v_t.shape[0])] = np.diag(s)  # fill sigma with EV
    # print(u.shape, s.shape, s_filled.shape, v_t.shape)
    u_ = np.dot(u, s_filled)  # multiply with singular values matrix
    return u_, v_t


def score(result, asked):
    RMSE = 0.
    for (r, c, rating) in asked:
        RMSE += (result[r][c] - rating) ** 2
    RMSE /= len(asked)
    return RMSE ** 0.5


weights = [100, 10, 1, 10, 100]
loaded_matrix, given_ratings, means = load_data_movie_mean('../input/data_train.csv', weights)
U, V = compute_svd(loaded_matrix, 41)
asked_entries = get_asked_entries()
result_matrix = predict_values(asked_entries, U, V, loaded_matrix)
store_data(result_matrix)

# code for cross validation:
# weights = [100, 10, 1, 10, 100]
# loaded_matrix, given_ratings, means = load_data_movie_mean('../input/data_train.csv', weights)
# for k in range(50):
#     pred_matrix, asked_entries = split_data(loaded_matrix, given_ratings, means)
#     U, V = compute_svd(pred_matrix, k)
#     result_matrix = predict_values(asked_entries, U, V, pred_matrix)
#     print("k = " + str(k) + ": " + str(score(result_matrix, asked_entries)))
