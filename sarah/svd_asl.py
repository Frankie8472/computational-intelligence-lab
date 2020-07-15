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


# compute the regularized Frobenius loss
def compute_loss(data_mat, rating_list, lamb, U_mat, V_mat):
    frobenius_loss = 0
    for (r, c, star_rating) in rating_list:
        frobenius_loss += (data_mat[r, c] - (np.dot(U_mat[r, :], V_mat[:, c]))) ** 2

    u_norm = 0
    for i in range(10000):
        u_norm += np.linalg.norm(U_mat[i, :]) ** 2
    frobenius_loss += u_norm * lamb

    v_norm = 0
    for j in range(1000):
        v_norm += np.linalg.norm(V_mat[:, j]) ** 2
    frobenius_loss += v_norm * lamb

    return frobenius_loss


def replace_u(rating_list, data_mat, V_mat, lamb, k_val):
    new_u = np.full((k_val, k_val), 0.0)

    for (r, c, star_rating) in rating_list:
        new_u += np.outer(V_mat[:, c], V_mat[:, c])

    new_u = np.add(new_u, lamb * np.identity(k_val))

    new_u = np.linalg.inv(new_u)

    vec_sum = np.full((k_val, ), 0.0)
    for (r, c, star_rating) in rating_list:
        vec_sum += data_mat[r, c] * V_mat[:, c]

    new_u = np.matmul(new_u, vec_sum)

    return new_u


def replace_v(rating_list, data_mat, U_mat, lamb, k_val):
    new_v = np.full((k_val, k_val), 0.0)

    for (r, c, star_rating) in rating_list:
        new_v += np.outer(U_mat[r, :], U_mat[r, :])

    new_v = np.add(new_v, lamb * np.identity(k_val))

    new_v = np.linalg.inv(new_v)

    vec_sum = np.full((k_val, ), 0.0)
    for (r, c, star_rating) in rating_list:
        vec_sum += data_mat[r, c] * U_mat[r, :]

    new_v = np.matmul(new_v, vec_sum)

    return new_v


weights = [100, 10, 1, 10, 100]
loaded_matrix, given_ratings, means = load_data_movie_mean('../input/data_train.csv', weights)
U, V = compute_svd(loaded_matrix, 41)

# prepare the ratings
ratings_row = [[] for i in range(10000)]
ratings_col = [[] for j in range(1000)]

for (row, column, rating) in given_ratings:
    ratings_row[row].append((row, column, rating))
    ratings_col[column].append((row, column, rating))


# set parameter k and lambda
# k is the number of factors, lam is the regularization strength
k = 1000
lam = 50


# compute the initial loss
old_loss = compute_loss(loaded_matrix, given_ratings, lam, U, V)
loss = 0
print(old_loss)
while (abs(old_loss - loss) > 1):
    for i in range(10000):
        U[i, :] = replace_u(ratings_row[i], loaded_matrix, V, lam, k)

    for i in range(1000):
        V[:, i] = replace_v(ratings_col[i], loaded_matrix, U, lam, k)

    old_loss = loss
    loss = compute_loss(loaded_matrix, given_ratings, lam, U, V)
    print(loss)

result = np.matmul(U, V)

# reverse the previous data centering and store the data in the submission file
store_data_float(result, given_ratings)




