from franz.importer import import_data
import numpy as np
from scipy import sparse as sp


csr_matrix = import_data()

# (1, 1000) How many users have seen this movie?
usr_cnt = np.asmatrix(csr_matrix.getnnz(0))

# (10000, 1) How many movies has this user seen?
mov_cnt = np.asmatrix(csr_matrix.getnnz(1)).transpose()

# (1, 1000) Sum of all ratings of this movie
usr_sum = csr_matrix.sum(0)

# (10000, 1) Sum of all ratings of this user
mov_sum = csr_matrix.sum(1)

# print(np.shape(np.mean(csr_matrix, 0)))
# print(np.shape(np.mean(csr_matrix, 1)))
matrix = csr_matrix.todense()
matrix[matrix == 0] = np.nan

print(np.nanmean(matrix, 1))
# print(np.shape(np.var(csr_matrix, 1)))

# (1, 1000) Mean rating of all ratings of this movie

# print(np.shape(sp.h
# Jede Reihe dividiert durch geschaute filme
#print((sp.csr_matrix((csr_matrix.transpose() / np.asmatrix(csr_matrix.getnnz(0))).transpose())))
# print(np.shape(np.divide(csr_matrix.sum(1), np.asmatrix(csr_matrix.getnnz(1)).transpose())))  # Was für ein Bewertungsdruchschnitt hat der user
