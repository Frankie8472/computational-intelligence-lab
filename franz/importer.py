from scipy import sparse as sp


def import_data():
    rows = []
    cols = []
    data = []

    with open('../input/data_train.csv', 'r') as file:
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

    csr_matrix = sp.csr_matrix((data, (rows, cols)))

    return csr_matrix
