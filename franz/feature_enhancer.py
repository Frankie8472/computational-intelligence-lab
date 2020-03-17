from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import numpy as np


## USER FEATURES (10000, 1)
# Total movies seen by this user
def usr_mov_cnt(sparse_matrix: csr_matrix):
    return np.asmatrix(sparse_matrix.getnnz(1)).transpose()


# User rating sum
def usr_mov_sum(sparse_matrix: csr_matrix):
    return sparse_matrix.sum(1)


# User rating mean of all seen movies
def usr_mov_mean_no_zero(sparse_matrix: csr_matrix):
    return np.divide(usr_mov_sum(sparse_matrix), usr_mov_cnt(sparse_matrix))


# User rating mean of all available movies
def usr_mov_mean(sparse_matrix: csr_matrix):
    return sparse_matrix.mean(1)


# User rating var of all seen movies
def usr_mov_var_no_zero(sparse_matrix: csr_matrix):
    return usr_mov_mean_no_zero(sparse_matrix.power(2)) - np.square(usr_mov_mean_no_zero(sparse_matrix))


# User rating var of all available movies
def usr_mov_var(sparse_matrix: csr_matrix):
    return usr_mov_mean(sparse_matrix.power(2)) - np.square(usr_mov_mean(sparse_matrix))


## MOVIE FEATURES (1, 1000)
# Number of people who have seen this movie
def mov_usr_cnt(sparse_matrix: csr_matrix):
    return np.asmatrix(sparse_matrix.getnnz(0))


# Sum of all ratings of this movie
def mov_usr_sum(sparse_matrix: csr_matrix):
    return sparse_matrix.sum(0)


# Movie rating mean of all who have seen this movie
def mov_usr_mean_no_zero(sparse_matrix: csr_matrix):
    return np.divide(mov_usr_sum(sparse_matrix), mov_usr_cnt(sparse_matrix))


# Movie rating mean of all people
def mov_usr_mean(sparse_matrix: csr_matrix):
    return sparse_matrix.mean(0)


# Movie rating variance of all who have seen this movie
def mov_usr_var_no_zero(sparse_matrix: csr_matrix):
    return mov_usr_mean_no_zero(sparse_matrix.power(2)) - np.square(mov_usr_mean_no_zero(sparse_matrix))


# Movie rating variance of all people
def mov_usr_var(sparse_matrix: csr_matrix):
    return mov_usr_mean(sparse_matrix.power(2)) - np.square(mov_usr_mean(sparse_matrix))


# Calculate sparse svd
def sparse_svd(sparse_matrix: csr_matrix):
    return svds(sparse_matrix)
