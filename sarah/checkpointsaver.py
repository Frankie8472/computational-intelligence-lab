from sarah.inputhandler import *
from numpy import genfromtxt

U = genfromtxt('checkpointU.csv', delimiter=',')
V = genfromtxt('checkpointV.csv', delimiter=',')

result = np.matmul(U, V)

weights = [100, 10, 1, 10, 100]
loaded_matrix, given_ratings, means = load_data_movie_mean('../input/data_train.csv', weights)

# reverse the previous data centering and store the data in the submission file
store_data_float(reverse_centering_deviation(result, given_ratings))