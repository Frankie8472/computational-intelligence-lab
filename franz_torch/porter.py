import torch
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


def import_dataframe(filepath='../input/data_train.csv', column_names=None):
    if column_names is None:
        column_names = ['userId', 'movieId', 'rating']

    return pd.DataFrame(import_raw_data(filepath), columns=column_names)


def import_csr(filepath='../input/data_train.csv'):
    data = import_raw_data(filepath)
    return sp.csr_matrix((data[:, 2], (data[:, 0], data[:, 1])))


def import_coo(filepath='../input/data_train.csv'):
    data = import_raw_data(filepath)
    return sp.coo_matrix((data[:, 2], (data[:, 0], data[:, 1])))


def csv_to_pt(filepath='../input/data_train.csv'):
    data = import_coo(filepath)
    indices = torch.tensor(np.vstack((data.row, data.col)))
    values = torch.tensor(np)
    tensor = torch.sparse_coo_tensor(indices=indices, values=values, dtype=torch.int8).to_dense()
    torch.save(tensor, filepath[0:-3] + "pt")
    return


def export_data(data, name):
    with open('../output/' + name + 'Submission.csv', 'w') as file:
        file.write('Id,Prediction')
        i = 0
        nnz = data.nonzero()
        nnz = nnz.cpu()
        nnz = nnz.numpy()
        nnz = nnz[nnz[:, 0].argsort()]
        nnz = nnz[nnz[:, 1].argsort(kind='mergesort')]

        for idx in nnz:
            row = idx[0]
            col = idx[1]
            val = data.data[row, col].item()
            file.write("\n")
            file.write('r' + str(row + 1) + '_c' + str(col + 1) + ',' + str(round(val)))
            i += 1
            if i % 10e4 == 0:
                print(i)
    return
