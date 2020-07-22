from sarah.inputhandler import *


# regularizer for the U and V matrices
reg = 0.08

# regularizer for the biasU and biasV matrices
reg2 = 0.04

# number of features
k_arr = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

# learning factors
lr_arr = [[0.1, 0.1/2, 0.1/4, 0.1/8, 0.1/16, 0.1/32, 0.1/64, 0.1/128, 0.1/256, 0.1/512],
          [0.1, 0.066666667, 0.05, 0.04, 0.033333333, 0.028571429, 0.025, 0.022222222, 0.02, 0.018181818],
          [0.1, 0.095238095, 0.090909091, 0.086956522, 0.083333333, 0.08, 0.076923077, 0.074074074, 0.071428571, 0.068965517],
          [0.1, 0.099502488, 0.099009901, 0.098522167, 0.098039216, 0.097560976, 0.097087379, 0.096618357, 0.096153846, 0.09569378]]

nr_users = 10000
nr_movies = 1000

results = np.zeros((nr_users, nr_movies))

for k in k_arr:
    for lrs in lr_arr:
        # number of iterations
        it = 100000000

        # load data
        data = load_data_raw()
        global_mean = np.mean(data[2])
        data = list(zip(data[1], data[0], data[2]))

        U = np.random.uniform(0, 0.05, (nr_users, k))
        V = np.random.uniform(0, 0.05, (nr_movies, k))

        biasU = np.zeros(nr_users)
        biasV = np.zeros(nr_movies)

        index = -1

        for s in range(it):
            if s % 10000000 == 0:
                index += 1
                lr = lrs[index]
                print(lr)

            movie, user, rating = random.choice(data)
            movie, user = movie-1, user-1 # because the data is 1-indexed

            U_movie = U[movie, :]
            V_user = V[user, :]

            biasU_movie = biasU[movie]
            biasV_user = biasV[user]

            prediction = U_movie.dot(V_user) + biasU_movie + biasV_user

            # error
            delta = rating - prediction

            # update U and V matrices
            new_U_movie = U_movie + lr * (delta * V_user - reg * U_movie)
            new_V_user = V_user + lr * (delta * U_movie - reg * V_user)
            # update biases
            new_biasU_movie = biasU_movie + lr * (delta - reg2 * (biasU_movie + biasV_user - global_mean))
            new_biasV_user = biasV_user + lr * (delta - reg2 * (biasV_user + biasU_movie - global_mean))

            U[movie, :] = new_U_movie
            V[user, :] = new_V_user

            biasU[movie] = new_biasU_movie
            biasV[user] = new_biasV_user

        pred = U.dot(V.T) + biasU.reshape(-1, 1) + biasV
        pred[pred > 5.0] = 5.0
        pred[pred < 1.0] = 1.0
        print(pred)
        results += pred

store_data_float(np.divide(results, len(k_arr) * len(lr_arr)))
