########################################################################################################################
# This is the simple baseline approach from the exercise sheet 2, where every missing value is replaced with the mean  #
# of all observed values. This scores 1.41035                                                                          #
########################################################################################################################

# movie rating prediction based on given matrix entries
# 10'000 Users (rows)
# 1'000 Movies (columns)

import numpy as np
from sarah.inputhandler import *

loaded_matrix, ratings, data_mean = load_data('../input/data_train.csv')
store_data(loaded_matrix)
