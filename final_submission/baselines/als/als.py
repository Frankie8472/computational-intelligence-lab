from typing import *
import random
import numpy as np
import als_data_util as util


# compute the regularized Frobenius loss
def compute_loss(data_mat: np.ndarray,
        rating_list: List[Tuple[int, int, int]], lamb: int,
        U_mat: np.ndarray, V_mat: np.ndarray) -> float:
    '''
    Compute the frobenius loss.
    :param data_mat: rating matrix containing known ratings
    :param rating_list: list of tuples (row, column, rating) of known
        ratings
    :param lamb: regularization strength
    :param U_mat: matrix U
    :param V_mat: matrix V
    :returns: frobenius loss
    '''
    frobenius_loss = 0
    for (r, c, star_rating) in rating_list:
        frobenius_loss += (data_mat[r, c] - (np.dot(U_mat[r, :],
            V_mat[:, c]))) ** 2

    u_norm = 0
    for i in range(10000):
        u_norm += np.linalg.norm(U_mat[i, :]) ** 2
    frobenius_loss += u_norm * lamb

    v_norm = 0
    for j in range(1000):
        v_norm += np.linalg.norm(V_mat[:, j]) ** 2
    frobenius_loss += v_norm * lamb

    return frobenius_loss


def replace(rating_list: List[Tuple[int, int, int]],
        data_mat: np.ndarray, U_mat: np.ndarray, lamb: int, k_val: int,
        v: bool) -> np.ndarray:
    '''
        Update a vector in matrix U or V as described in the ALS algorithm.

        :param rating_list: list of tuples (row, column, rating) of known
            ratings
        :param data_mat: rating matrix containing known ratings
        :param U_mat: matrix U or V which should be updated
        :param lamb: regularization strength
        :param k_val: number of factors
        :param v: True if matrix V should be updated, False otherwise
        :returns: updated vector
    '''
    new_v = np.full((k_val, k_val), 0.0)

    for (r, c, star_rating) in rating_list:
        if v:
            new_v += np.outer(U_mat[r, :], U_mat[r, :])
        else:
            new_v += np.outer(U_mat[:, c], U_mat[:, c])

    new_v = np.add(new_v, lamb * np.identity(k_val))

    new_v = np.linalg.inv(new_v)

    vec_sum = np.full((k_val, ), 0.0)
    for (r, c, star_rating) in rating_list:
        if v:
            vec_sum += data_mat[r, c] * U_mat[r, :]
        else:
            vec_sum += data_mat[r, c] * U_mat[:, c]

    new_v = np.matmul(new_v, vec_sum)

    return new_v


def main() -> None:
    '''
    Do Alternating Least Squares.
    '''
    # load the data
    data, ratings, means = util.load_data_movie_mean(
            '../../input/data_train.csv', [100, 10, 1, 10, 100])

    # center the data to remove bias
    data = util.center_deviation_movie_mean(data, ratings)

    # prepare the ratings
    ratings_row = [[] for i in range(10000)]
    ratings_col = [[] for j in range(1000)]

    for (row, column, rating) in ratings:
        ratings_row[row].append((row, column, rating))
        ratings_col[column].append((row, column, rating))

    # set parameter k and lambda
    # k is the number of factors, lam is the regularization strength
    k = 41
    lam = 50

    # Initialize U and V
    U = np.full((10000, k), 0.0)
    V = np.full((k, 1000), 0.0)

    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            U[i][j] = random.random()

    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            V[i][j] = random.random()

    # compute the initial loss
    old_loss = compute_loss(data, ratings, lam, U, V)
    loss = 0
    print(old_loss)
    # do alternating least squares
    while (abs(old_loss - loss) > 1):
        for i in range(10000):
            U[i, :] = replace(ratings_row[i], data, V, lam, k, False)

        for i in range(1000):
            V[:, i] = replace(ratings_col[i], data, U, lam, k, True)

        old_loss = loss
        loss = compute_loss(data, ratings, lam, U, V)
        print(loss)

    # compute prediction matrix
    result = np.matmul(U, V)

    # reverse the previous data centering and store the data in the
    # submission file
    output_path = 'output/als_prediction.csv'
    util.store_data_float(util.reverse_centering_deviation(result,ratings),
            output_path)


if __name__ == "__main__":
    main()
