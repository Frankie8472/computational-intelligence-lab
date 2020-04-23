import numpy as np
import pandas as pd
from scipy import sparse as sp


def import_raw_data(filepath):
    rows = []
    cols = []
    data = []

    with open(filepath, 'r') as file:
        header = file.readline()

        for line in file:
            entry, prediction = line.split(',')
            star_rating = int(prediction)
            row_entry, column_entry = entry.split('_')
            row = int(row_entry[1:]) - 1
            col = int(column_entry[1:]) - 1
            rows.append(row)
            cols.append(col)
            data.append(star_rating)

    return np.c_[rows, cols, data]


def import_dataframe():
    filepath = '../input/data_train.csv'
    column_names = ['userId', 'movieId', 'rating']
    return pd.DataFrame(import_raw_data(filepath), columns=column_names, dtype=np.float64)


def import_csr():
    filepath = '../input/data_train.csv'
    data = import_raw_data(filepath)
    return sp.csr_matrix((data[2], (data[0], data[1])), dtype=np.float64)


def export_data(extended_matrix, name):
    filepath = '../output/sampleSubmission.csv'
    data = import_raw_data(filepath)

    with open('../output/' + name + 'Submission.csv', 'w') as file:
        file.write('Id,Prediction\n')
        for i in range(0, len(data[0])):
            row = data[0][i] + 1
            col = data[1][i] + 1
            file.write('r' + str(row) + '_c' + str(col) + ',' + str(round(extended_matrix[row - 1, col - 1])))
            file.write("\n")
    return
