from inputhandler_sla import *


# regularizer for the U and V matrices
reg = 0.08

# regularizer for the biasU and biasV matrices
reg2 = 0.04

# number of features
k_arr = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

nr_users = 10000
nr_movies = 1000

result_avg = np.zeros((nr_users, nr_movies))

for k in k_arr:
    SGD_result = run_SGD(k)
    result = SGD_result[0]
    result_avg += result
    validation_set = SGD_result[1]
    # cross validate
    score = SGD_cross_validate(result, validation_set)
    print('SGD with k = ' + str(k) + ', score = ' + str(score))
    output_path = 'output_data/SGD_k'+str(k)+'.csv'
    store_data_float(result, output_path)
    print('stored in: '+str(output_path))

result_avg = np.divide(result_avg, len(k_arr))
output_path = 'output_data/SGD_average_k_'+str(k_arr[0])+'-'+str(k_arr[len(k_arr)-a])+'.csv'
store_data_float(result_avg, output_path)
print('SGD averaged for k =' + str(k)+ ' stored in: '+str(output_path))


def run_SGD(k: int) -> ('numpy.ndarray', tuple):
   # number of iterations
    it = 100000000

    # load data
    data = load_data_raw()

    global_mean = np.mean(data[2])
    data = list(zip(data[1], data[0], data[2]))
    validation_set = list(zip(data[3], data[4], data[5]))

    U = np.random.uniform(0, 0.05, (nr_users, k))
    V = np.random.uniform(0, 0.05, (nr_movies, k))

    biasU = np.zeros(nr_users)
    biasV = np.zeros(nr_movies)

    lr = 0.1
    for s in range(it):
        if s % 10000000 == 0:
            print(lr)
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
        new_biasU_d = biasU_d + lr * (delta - reg2 * (biasU_d + biasV_n - global_mean))
        new_biasV_n = biasV_n + lr * (delta - reg2 * (biasV_n + biasU_d - global_mean))

        U[d, :] = new_U_d
        V[n, :] = new_V_n

        biasU[d] = new_biasU_d
        biasV[n] = new_biasV_n

    pred = U.dot(V.T) + biasU.reshape(-1, 1) + biasV
    pred[pred > 5.0] = 5.0
    pred[pred < 1.0] = 1.0
    print(pred)
    return pred, validation_set


