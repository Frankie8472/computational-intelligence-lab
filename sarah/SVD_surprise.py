from surprise import prediction_algorithms
from sarah.inputhandler import *

data = load_data_surprise()

trainset = data.build_full_trainset()

# use the SVD++ algorithm from the surprise library
algo = prediction_algorithms.matrix_factorization.SVDpp(n_factors=12, lr_all=0.085,
                                                        n_epochs=50, reg_all=0.01, verbose=True)

# Run 5-fold cross-validation and print results.
# cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)

algo.fit(trainset)

file = open('SarahSubmission.csv', 'w+')  # open a new file to write into
file.write('Id,Prediction\n')  # the header line
asked_entries = get_asked_entries()
for (i, j) in asked_entries:
    file.write('r' + str(i+1) + '_c' + str(j+1) + ',' + str(algo.predict(i, j, verbose=True).est) + '\n')
