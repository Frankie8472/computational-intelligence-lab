from surprise.model_selection import cross_validate
from surprise import KNNWithMeans
from sarah.inputhandler import *

# Step 1 - Data Import & Preparation

data = load_data_surprise()


trainsetfull = data.build_full_trainset()
print('Number of users: ', trainsetfull.n_users, '\n')
print('Number of items: ', trainsetfull.n_items, '\n')

# Step 2 - Cross-Validation

my_k = 15
my_min_k = 5
my_sim_option = {
    'name': 'pearson', 'user_based': False
}

algo = KNNWithMeans(
    k=my_k, min_k=my_min_k,
    sim_options=my_sim_option, verbose=True
)

results = cross_validate(
    algo=algo, data=data, measures=['RMSE'],
    cv=5, return_train_measures=True
)

print(results['test_rmse'].mean())

# Step 3 - Model Fitting

algo.fit(trainsetfull)

# Step 4 - Prediction

file = open('SarahSubmission.csv', 'w+')  # open a new file to write into
file.write('Id,Prediction\n')  # the header line
asked_entries = get_asked_entries()
for (i, j) in asked_entries:
    file.write('r' + str(i+1) + '_c' + str(j+1) + ',' + str(algo.predict(i, j, verbose=True).est) + '\n')
