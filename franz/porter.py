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

    csr_matrix = sp.csr_matrix((data, (rows, cols)), dtype=float)

    return csr_matrix


def export_data(extended_matrix, name):
    rows = []
    cols = []
    data = []

    with open('../output/sampleSubmission.csv', 'r') as file:
        file.readline()

        for line in file:
            entry, prediction = line.split(',')
            star_rating = int(prediction)
            row_entry, column_entry = entry.split('_')
            row = int(row_entry[1:]) - 1
            col = int(column_entry[1:]) - 1
            rows.append(row)
            cols.append(col)
            data.append(star_rating)

    with open('../output/'+name+'Submission.csv', 'w') as file:
        file.write('Id,Prediction\n')
        for i in range(0, len(rows)):
            row = rows[i] + 1
            col = cols[i] + 1
            file.write('r'+str(row)+'_c'+str(col)+','+str(round(extended_matrix[row-1, col-1])))
            file.write("\n")

    return
