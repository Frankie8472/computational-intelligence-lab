import numpy as np
from matplotlib import pyplot as plt
'''
STEP 2: Improve unknown ratings. Do not change known ratings!
Using cross validation, we compare the score of different approaches to each
other and to the score before improving the prediction.
'''

def run(A: 'numpy.ndarray', ratings: set, asked_ratings: set, users: int,
        movies: int, approach_id) -> 'numpy.ndarray':
    '''
    This function is called by run.py to improve rating predictions.

    :param A: full input rating matrix 
    :param ratings: set of known rating tuples (row, column, rating)
    :param asked_ratings: subset of ratings for which the rating is asked
        for in cross validation
    :param users: number of users
    :param movies: number of movies
    :param approach_id: ID of the approach which should be used
    :return: full rating matrix
    '''
    return do_improve[approach_id](A, ratings, asked_ratings)


def svd(A: 'numpy.ndarray', ratings: set,
        asked_ratings: set) -> 'numpy.ndarray':
    '''
    Does SVD on matrix A.

    :param A: full input rating matrix
    :param ratings: set of known rating tuples (row, column, rating)
    :param asked_ratings: subset of ratings for which the rating is asked
        for in cross validation
    :return: full rating matrix
    '''
    print('Prediction improvement approach: svd')
    U, d, Vt = np.linalg.svd(A, full_matrices=True)
    D = np.zeros(A.shape)
    D[:A.shape[1], :A.shape[1]] = np.diag(d)
    k = 13
    Dk = D.copy()
    Dk[k:, k:] = 0
    Ak = U.dot(Dk).dot(Vt)
    return Ak


def error_adjustment(A, ratings, asked_ratings):
    '''
    Factorize input matrix A = XY. Repeat: For each entry in A where
    cross validation asks for a prediction, calculate the
    error = actual - prediction and adjust the values in XY accrodingly.
    This approach is similar to gradient descend. However, the approach
    minimizes the error linearly and uses the error instead of a gradient.

    :param A: full input rating matrix
    :param ratings: set of known rating tuples (row, column, rating)
    :param asked_ratings: subset of ratings for which the rating is asked
        for in cross validation
    :return: full rating matrix
    '''
    print('Prediction improvement approach: error adjustment')
    A_known = np.zeros(A.shape)
    for (r, c, rating) in ratings:
        if not (r, c, rating) in asked_ratings:
            A_known[r,c] = rating
    # Factorize A = (U*D)*Vt = X*Y
    U, d, Vt = np.linalg.svd(A_known, full_matrices=True)
    D = np.zeros(A.shape)
    D[:A.shape[1], :A.shape[1]] = np.diag(d)
    X = U.dot(D)
    Y = Vt
    





do_improve = {  1: svd,
                2: error_adjustment}
