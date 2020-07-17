import csv
import numpy as np
import re
import matplotlib.pyplot as plt

data = np.full((10000, 1000), 3) # make a matrix only with 3's
mask = np.full((10000, 1000), True) # make a 'True' matrix

with open('../input/data_train.csv') as file:
    reader = csv.reader(file, delimiter=',')
    pattern = "\D+"
    next(reader)
    # mark all given entries as False in the mask
    # fill every given value in the data matrix
    for row in reader:
        positions = re.split(pattern, row[0])
        mask[int(positions[1]) - 1][int(positions[2]) - 1] = False
        data[int(positions[1]) - 1][int(positions[2]) - 1] = int(row[1])

# print(mask)
lambda_ = 0.1
n_factors = 10
m, n = data.shape # m is the number of users, n is the number of movies
n_iterations = 10
W = mask.copy() # W is the same as the mask
print(np.dot(W[0], data[0])) # prints 3 * the sum of all missing values
W[W == True] = 0 # make the whole matrix False
W[W == False] = 1 # make the whole matrix True

# do something something with data
masked_data = np.ma.array(data, mask=mask) # mark all given entries as valid
average_user_rating = masked_data.mean(axis=1) # compute the average rating per user
correction = average_user_rating - np.full((len(average_user_rating)), 3) # subtract the mean
masked_data = (masked_data.T - correction).T # subtract the deviation from the mean from every column
mean = masked_data.mean(axis=0) # compute the mean per movie
normalized = masked_data - mean # subtract the movie_mean
u, s, vh = np.linalg.svd(normalized.filled(0)) # all masked values (not given) are filled with 0

smat = np.zeros((10000, 1000)) # make a matrix filled with zeros
k = 10
u_modify = np.zeros((10000, k)) # make a matrix filled with zeros
u_modify[:k, :k] = np.diag(np.sqrt(s[:k])) # fill with the square root of the first k singular values
v_modify = np.zeros((k, 1000)) # make a matrix filled with zeros
v_modify[:k, :k] = np.diag(np.sqrt(s[:k])) # fill with the square root of the first k singular values


def get_error(Q, X, Y, W):
    return np.sum((W * (Q - np.dot(X, Y))) ** 2)


# turn into two matrices
X = np.dot(u, u_modify)
Y = np.dot(v_modify, vh)
Q = data

######################################## Simple SVD up to here

# do als

errors = []
errors.append(get_error(Q, X, Y, W))
for ii in range(n_iterations):
    print(ii)
    for u, Wu in enumerate(W):
        X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(n_factors),
                               np.dot(Y, np.dot(np.diag(Wu), Q[u].T))).T
    for i, Wi in enumerate(W.T):
        Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(n_factors),
                                 np.dot(X.T, np.dot(np.diag(Wi), Q[:, i])))
    if ii % 2 == 0:
        print('{}th iteration is completed'.format(ii))
    errors.append(get_error(Q, X, Y, W))

plt.plot(errors)
plt.show()
reconstructed = np.dot(X, Y)
print(X)
print(Y)
print(reconstructed)
data = reconstructed + mean
data = (data.T + correction).T

with open('../output/sampleSubmission.csv') as template:
    reader = csv.reader(template, delimiter=',')
    next(reader)
    with open('SarahSubmission.csv', 'w', newline='') as solution:
        writer = csv.writer(solution, delimiter=',')
        writer.writerow(['Id', 'Prediction'])
        for row in reader:
            positions = re.split(pattern, row[0])
            value = data[int(positions[1]) - 1][int(positions[2]) - 1]
            if value > 5:
                value = 5
            if value < 1:
                value = 1
            # print(int(value))
            writer.writerow([row[0], value])
