import numpy as np
import random
import math

import data_util as util
'''
STEP 2: Improve unknown ratings.
Using cross validation, we compare the score of different approaches to each
other and to the score before improving the prediction.
'''

def run(A: 'numpy.ndarray', ratings: set, asked_ratings: set, users: int,
        movies: int, approach_id, svd_k: int = 0) -> 'numpy.ndarray':
    '''
    This function is called by run.py to improve rating predictions.

    :param A: full input rating matrix 
    :param ratings: set of known rating tuples (row, column, rating)
    :param asked_ratings: subset of ratings for which the rating is asked
        for in cross validation
    :param users: number of users
    :param movies: number of movies
    :param approach_id: ID of the approach which should be used
    :param svd_k: number of eigenvalues to keep for SVD
    :return: full rating matrix
    '''
    return do_improve[approach_id](A, ratings, asked_ratings, svd_k)


def svd(A: 'numpy.ndarray', ratings: set,
        asked_ratings: set, k: int) -> 'numpy.ndarray':
    '''
    Does SVD on matrix A.

    :param A: full input rating matrix
    :param ratings: set of known rating tuples (row, column, rating)
    :param asked_ratings: subset of ratings for which the rating is asked
        for in cross validation
    :param k: number of eigenvalues to keep
    :return: full rating matrix
    '''
    print('Prediction improvement approach: SVD')
    U, d, Vt = np.linalg.svd(A, full_matrices=True)
    D = np.zeros(A.shape)
    D[:A.shape[1], :A.shape[1]] = np.diag(d)
    Dk = D.copy()
    Dk[k:, k:] = 0
    Ak = U.dot(Dk).dot(Vt)
    return Ak


do_improve = {1: svd}
