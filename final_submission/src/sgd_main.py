import numpy as np
import random
import time
import sgd_data_util as util
from multiprocessing import Pool
from typing import *


def run_SGD(k: int, reg: float,
        reg2: float) -> ('numpy.ndarray', List[Tuple[int, int, int]]):
    '''
    Does SGD with given k, and regularizers reg and reg2.

    :param k: number of features
    :param reg: regularizer for matrices U and V
    :param reg2: regularizer for matrices biasU and biasV
    :returns:
        - predicted rating matrix A
        - validation set for cross validation which is  a list of
            tuples (r, c, rating), s.t. A[r][c] = rating
    '''
    # number of iterations
    it = 100000000

    # load data
    data_all = util.load_data_raw()
    data = list(zip(data_all[0], data_all[1], data_all[2]))
    validation_set = list(zip(data_all[3], data_all[4], data_all[5]))

    global_mean = np.mean(data_all[2])

    U = np.random.uniform(0, 0.05, (nr_users, k))
    V = np.random.uniform(0, 0.05, (nr_movies, k))

    biasU = np.zeros(nr_users)
    biasV = np.zeros(nr_movies)

    lr = 0.1

    for s in range(it):
        if s % 10000000 == 0:
            # print('k='+str(k)+', learning rate: '+str(lr))
            lr /= 2
        d, n, v = random.choice(data)
        d, n = d-1, n-1
        U_d = U[d, :]
        V_n = V[n, :]

        biasU_d = biasU[d]
        biasV_n = biasV[n]

        guess = U_d.dot(V_n) + biasU_d + biasV_n

        # error
        delta = v - guess

        # update U and V matrices
        new_U_d = U_d + lr * (delta * V_n - reg * U_d)
        new_V_n = V_n + lr * (delta * U_d - reg * V_n)
        # update biases
        new_biasU_d = biasU_d + lr * (delta - reg2 *
                (biasU_d + biasV_n - global_mean))
        new_biasV_n = biasV_n + lr * (delta - reg2 *
                (biasV_n + biasU_d - global_mean))

        U[d, :] = new_U_d
        V[n, :] = new_V_n

        biasU[d] = new_biasU_d
        biasV[n] = new_biasV_n
 
    pred = U.dot(V.T) + biasU.reshape(-1, 1) + biasV
    pred[pred > 5.0] = 5.0
    pred[pred < 1.0] = 1.0
    
    return pred, validation_set


def run_parallel(k_r_r2: Tuple[int, float, float]) -> 'numpy.ndarray':
    '''
    Does SGD with given k and regularizers r and r2. Stores predicted values
    in a file. Outputs score (RMSE) using cross validation, execution time,
    and output path relative to the current folder.

    :param k_r_r2: tuple (k, r, r2), where k is the number of features used
        in SGD, r the regularizer for matrices U and V, and r2 the
        regularizer for matrices biasU and biasV
    :returns: predicted rating matrix
    '''
    print('>>> Running SGD with k_r_r2 = '+str(k_r_r2))
    start_time = time.time()

    k = k_r_r2[0]
    reg = k_r_r2[1]
    reg2 = k_r_r2[2]
    SGD_result = run_SGD(k, reg, reg2) 
    result = SGD_result[0]
    validation_set = SGD_result[1]

    # cross validate
    score = util.SGD_cross_validate(result, validation_set)

    # calculate execution time
    end_time = time.time()
    hours, remaining = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(remaining, 60)
    
    print('......................'+
            '\nSGD with k = ' + str(k) + ', reg = ' + str(reg) +
            ', reg2 = ' + str(reg2) + ', score = ' + str(score) +
            '\nExecution time: '+
            '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds))

    return result


def main() -> None:
    '''
    Does SGD using different k values
    '''
    # Grid search values:
    # number of features
    k_arr = [
            8 ,9, 10, 11,
            12, 13, 14, 15,
            16, 17]
    # regularizers should be in [0.01, 0.1], so adjust range accordingly
    # for U and V:
    reg_arr = [0.09]
    # for biasU and biasV
    reg2_arr = [0.05]
    # create list of all possible k, reg, and reg2 combinations
    combinations = [(k, r, r2) for k in k_arr for r in reg_arr
            for r2 in reg2_arr]

    # for each k, do SGD
    pool = Pool()
    results = pool.map(run_parallel, combinations)
    pool.close()
    pool.join()

    # average all obtained prediction matrices
    result_avg = np.zeros((nr_users, nr_movies))
    for r in results:
        result_avg += r
    result_avg = np.divide(result_avg, len(k_arr))

    # write output file
    output_path = '../output/SGD_submission.csv')
    util.store_data_float(result_avg, output_path)

    print(  '......................'+
            '\nSGD averaged for k =' + str(k_arr)+
            ' and regularizers' + str(reg_arr) + str(reg2_arr) +
            '\nStored in: '+str(output_path))

    return


if __name__ == "__main__":
    nr_users = 10000
    nr_movies = 1000
    
    main()
