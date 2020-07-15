from sarah.inputhandler import *
import sys


def cosine_similarity(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def pearson_correlation_coefficient(a, b):
    return np.inner(a - np.mean(a), b - np.mean(b)) / (np.linalg.norm(a - np.mean(a)) * np.linalg.norm(b - np.mean(b)))


def inverse_euclidean_distance(a, b):
    return 1 / np.linalg.norm(a - b)


def find_max(a):
    max = 0
    max_i = -1
    for i in range(len(a)):
        if max < a[i]:
            max = a[i]
            max_i = i
    return max, max_i


data, ratings = load_data_zeros('../input/data_train.csv')

result = data

distances = np.full((1000, 1000), sys.maxsize)

# number of neighbors
k = 200

# calculates the distance between every pair of movie-vectors
for i in range(1000):
    print(i)
    for j in range(i+1, 1000):
        distances[i, j] = inverse_euclidean_distance(data[:, i], data[:, j])
        distances[j, i] = data[i, j]


asked = get_asked_entries()

for (row, col) in asked:
    neighbor_indices = []
    neighbor_values = []
    sum = 0
    current_max = sys.maxsize
    max_index = -1
    for i in range(1000):
        if data[row, i] != 0 and len(neighbor_indices) < k:
            neighbor_indices += [i]
            neighbor_values += [data[row, i]]
            sum += data[row, i]
            current_max, max_index = find_max(neighbor_values)

        elif data[row, i] != 0 and distances[col, i] < current_max:
            sum -= neighbor_values[max_index]
            neighbor_indices[max_index] = i
            neighbor_values[max_index] = data[row, i]
            sum += data[row, i]
            current_max, max_index = find_max(neighbor_values)

    result[row, col] = sum / k
    print(row)

store_data(result)
