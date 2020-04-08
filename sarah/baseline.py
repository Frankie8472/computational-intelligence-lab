########################################################################################################################
# This is the simple baseline approach from the exercise sheet 2, where every missing value is replaced with the mean  #
# of all observed values.                                                                                              #
########################################################################################################################

# movie rating prediction based on given matrix entries
# 10'000 Users (rows)
# 1'000 Movies (columns)

import numpy as np

nr_of_users = 10000
nr_of_movies = 1000

ratings = []


def load_data(filename):
    nr_of_entries = 0
    sum_of_ratings = 0
    with open(filename, 'r') as file:
        file.readline()  # remove the header, don't save it, because we don't need it
        for line in file:
            entry, prediction = line.split(',')  # split the movie rating from the ID
            star_rating = int(prediction)  # save the rating
            row_entry, column_entry = entry.split('_')  # split the ID accordingly (they have the format rX_cY)
            row = int(row_entry[1:])  # remove the 'r'
            column = int(column_entry[1:])  # remove the 'c'
            ratings.append((row-1, column-1, star_rating))  # the IDs are 1-indexed, so subtract 1
            nr_of_entries += 1  # count how many entries there are for the mean calculation
            sum_of_ratings += star_rating  # for calculating the mean
    data_mean = sum_of_ratings / nr_of_entries  # calculate the mean
    print('the data mean is: ' + str(data_mean))
    data_matrix = np.full((nr_of_users, nr_of_movies), int(data_mean))  # fill every entry with the mean per default
    for (row, column, starRating) in ratings:
        data_matrix[row, column] = starRating  # replace the given entries with the ratings
    return data_matrix


def store_data(result_matrix):
    file = open('SarahSubmission.csv', 'w+')  # open a new file to write into
    file.write('Id, Prediction\n')  # the header line
    with open('sampleSubmission.csv', 'r') as sample_file:  # open the sample file to see which values I have to submit
        sample_file.readline()  # throw the header away
        for line in sample_file:
            entry, prediction = line.split(',')
            row_entry, column_entry = entry.split('_')
            row = int(row_entry[1:]) - 1
            column = int(column_entry[1:]) - 1
            file.write(entry + ',' + str(result_matrix[row][column]) + '\n')  # store with the same ID as the sample


loaded_matrix = load_data('data_train.csv')
store_data(loaded_matrix)
