import numpy as np
import random
import math
'''
STEP 2: Improve unknown ratings. Do not change known ratings!
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
    print('Prediction improvement approach: svd')
    U, d, Vt = np.linalg.svd(A, full_matrices=True)
    D = np.zeros(A.shape)
    D[:A.shape[1], :A.shape[1]] = np.diag(d)
    Dk = D.copy()
    Dk[k:, k:] = 0
    Ak = U.dot(Dk).dot(Vt)
    return Ak

weight = 0.2

def error_adjustment(A: 'numpy.ndarray', ratings: set, asked_ratings: set,
        k: int) -> 'numpy.ndarray':
    '''
    Factorize input matrix A = XY. Repeat: For each known rating in A where
    cross validation does not ask for a prediction, calculate the
    error = actual - prediction and adjust the values in XY accrodingly.
    This approach is similar to gradient descend. However, the approach
    minimizes the error linearly and uses the error instead of a gradient.

    :param A: full input rating matrix
    :param ratings: set of known rating tuples (row, column, rating)
    :param asked_ratings: subset of ratings for which the rating is asked
        for in cross validation
    :param k: not used
    :return: full rating matrix
    '''
    print('Prediction improvement approach: error adjustment')
    # Factorize A = (U*D)*Vt = X*Y
    U, d, Vt = np.linalg.svd(A, full_matrices=True)
    D = np.zeros(A.shape)
    D[:A.shape[1], :A.shape[1]] = np.diag(d)
    X = U.dot(D)
    Y = Vt
    # Create list of tuples (row, column, rating) of known ratings
    known = list(ratings.copy() - set(asked_ratings))
    # Adjust
    random.shuffle(known) 
    A = error_adjustment_one_cycle(X,Y,known)


def error_adjustment_one_cycle(X,Y,known):
    count = len(known)
    for (r, c, rating) in known:
        X[r,:], Y[:,c] = error_adjustment_one_iteration(X[r,:], Y[:,c],
                rating)
        count -= 1
        if count % 1000 == 0:
            print(count)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(Y))
    return X.dot(Y)


def error_adjustment_one_iteration(x, y, actual_rating):
    error = actual_rating - (x.dot(y))
    if np.isnan(error):
        '''
        print(type(x))
        print(type(y))
        print(np.shape(x))
        print(np.shape(y))
        print(actual_rating)
        print(np.any(np.isinf(x)))
        print(np.any(np.isnan(x)))
        print(np.any(np.isinf(y)))
        print(np.any(np.isnan(y)))
        print(x.dot(y))
        print(np.dot(x,y))
        print(np.dot(np.ravel(x), np.ravel(y)))
        print(np.dot(x,y.T))
        '''
        return x, y
    assert(not np.isnan(error)) 
    if bool(random.getrandbits(1)):
        if not np.any(np.isinf(x + error * weight)):
            x = x + error * weight
        error = actual_rating - (x.dot(y))
        if not np.any(np.isinf(y + error * weight)):
            y = y + error * weight
    else:
        if not np.any(np.isinf(y + error * weight)):
            y = y + error * weight
        error = actual_rating - (x.dot(y))
        if not np.any(np.isinf(x + error * weight)):
            x = x + error * weight
        #assert not np.any(np.isinf(x)) and not np.any(np.isinf(y))
    return x, y
    

do_improve = {  1: svd,
                2: error_adjustment}
