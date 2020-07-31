"""
This file is our baseline for the KNN algorithm
"""

from math import sqrt
import pandas as pd
import numpy as np
from multiprocessing import Pool

"""
The amount of neighbors to extract
"""
K = 50


def import_data(path):
    """
    Imports the data to work on
    :param path: Path where to find the data csv
    :return: The imported data
    """
    import_data_df = pd.read_csv(path, low_memory=False)
    import_data_df[['userId', 'movieId']] = import_data_df['Id'].str.split(
        "_", n=1, expand=True)
    import_data_df[['userId']] = import_data_df['userId'].str.strip("r")
    import_data_df[['movieId']] = import_data_df['movieId'].str.strip("c")
    import_data_df['rating'] = import_data_df['Prediction']
    import_data_df = import_data_df[['userId', 'movieId', 'rating']].astype(np.long)

    return pd.DataFrame(import_data_df)


def export_predictions(base, predictions, name):
    """
    Export the predictions to a file. It will overwrite any existing file
    :param base: The entries that should be exported. This should also contain a backup in case the prediction data does
                    not contain the prediction for a given entry.
    :param predictions: The predictions that have been generated.
    :param name: The name of the file to save. It will be stored as './output/' + name + 'Submission.csv'
    """
    no_ratings_count = 0
    with open('./output/' + name + 'Submission.csv', 'w') as file:
        file.write('Id,Prediction')
        for i in range(0, len(base)):
            row = int(base[i][0])
            col = int(base[i][1])
            val = base[i][2]
            df = predictions.get(row)
            if df is not None:
                try:
                    val = df.at[col, 'rating']
                except KeyError:
                    print("*WARNING*: No rating for user {} for movie {}".format(row, col))
                    no_ratings_count += 1
            if not (1 <= val <= 5):
                if not (0 <= val <= 6):
                    # Get rid of spam
                    print(
                        "WARNING: user {} for movie {} has invalid rating {}. Clamping.".format(row, col, val))
                val = min(5, max(1, val))
            file.write("\n")
            file.write('r' + str(row ) + '_c' + str(col) + ',' + str(val))
    if no_ratings_count:
        print("Had {} occurrences that could not be rated".format(no_ratings_count))


def get_input_user(data: pd.DataFrame):
    """
    Selects a user in the given data set and returns it. This is a generator function and
    can be iterated through.
    
    The user are sorted by the amount of movies that they have rated.

    :param data: Matrix containing the users and their ratings

    :returns: The input data, current user index, the ratings of the current user
    """
    sizes = data.groupby(['userId']).userId.agg('count').to_frame('c')
    sizes = sizes.sort_values(by="c", ascending=False).reset_index()
    for i in sizes["userId"]:
        yield data, i, data[data.userId == i]


def get_ratings(df, user_id, curr_user_data):
    """
    Calculate KNN of the selected user and predict the ratings.
    :param df: The data containing all ratings
    :param user_id: The user ID for which to predict the ratings
    :param curr_user_data: The data of the current user to work on
    :return: Prediction for the items for the given user
    """
    print("Using userId {}".format(user_id))

    # Get the users that have seen the same movies as the selected user
    user_subset = df[(df['movieId'].isin(curr_user_data['movieId'].tolist())) & (df.userId != user_id)]
    # Sort them by the amount of movies that they have in common with the selected user
    sizes = user_subset.groupby(['userId']).userId.agg('count').to_frame('c').reset_index()
    sizes = sizes.sort_values(by="c", ascending=False)[0:K * 2]

    # Get all users that are in the selected list
    user_subset = user_subset[user_subset['userId'].isin(sizes["userId"].tolist())]
    user_subset_group = user_subset.groupby(['userId'])

    # Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is
    # the coefficient
    pearson_correlation_dict = {}

    # Implementation based on: https://medium.com/swlh/how-to-build-simple-recommender-systems-in-python-647e5bcd78bd
    # For every user group in our subset
    for name, group in user_subset_group:
        # Let's start by sorting the input and current user group so the values aren't mixed up later on
        group = group.sort_values(by='movieId')
        input_movies = curr_user_data.sort_values(by='movieId')
        # Get the N for the formula
        n_ratings = len(group)
        # Get the review scores for the movies that they both have in common
        temp_df = input_movies[input_movies['movieId'].isin(group['movieId'].tolist())]
        # And then store them in a temporary buffer variable in a list format to facilitate future calculations
        temp_rating_list = temp_df['rating'].tolist()
        # Let's also put the current user group reviews in a list format
        temp_group_list = group['rating'].tolist()
        sum_rating = sum(temp_rating_list)
        sum_group = sum(temp_group_list)
        # Now let's calculate the pearson correlation between two users, so called, x and y
        # Sxx = sum(i ** 2 for i in temp_rating_list) - pow(sum_rating, 2) / float(n_ratings)
        # Syy = sum(i ** 2 for i in temp_group_list) - pow(sum_group, 2) / float(n_ratings)
        # Sxy = sum(i * j for i, j in zip(temp_rating_list, temp_group_list)) - sum_rating * sum_group / float(n_ratings)
        Sxx = sum([i ** 2 for i in temp_rating_list]) - pow(sum(temp_rating_list), 2) / float(n_ratings)
        Syy = sum([i ** 2 for i in temp_group_list]) - pow(sum(temp_group_list), 2) / float(n_ratings)
        Sxy = sum(i * j for i, j in zip(temp_rating_list, temp_group_list)) - sum(temp_rating_list) * sum(
            temp_group_list) / float(n_ratings)
        # If the denominator is different than zero, then divide, else, 0 correlation.
        if Sxx != 0 and Syy != 0:
            # Clamp to counter rounding errors
            pearson_correlation_dict[name] = min(1, max( -1, Sxy / sqrt(Sxx * Syy)))
        else:
            pearson_correlation_dict[name] = 0

    # Convert the dict to a DataFrame
    pearson_df = pd.DataFrame.from_dict(pearson_correlation_dict, orient='index')
    pearson_df.columns = ['similarityIndex']
    pearson_df['userId'] = pearson_df.index
    pearson_df.index = range(len(pearson_df))
    top_users = pearson_df.sort_values(by='similarityIndex', ascending=False)[0:K]
    
    # We only consider users with at least some similar taste i.e. similarityIndex > 0
    top_users = top_users[top_users["similarityIndex"] > 0]

    top_users_rating = top_users.merge(df, left_on='userId', right_on='userId', how='inner')

    # Multiplies the similarity by the user's ratings
    top_users_rating['weightedRating'] = top_users_rating['similarityIndex'] * top_users_rating['rating']

    # Applies a sum to the top_users after grouping it up by userId
    temp_top_users_rating = top_users_rating.groupby("movieId").sum()[["similarityIndex", "weightedRating"]]
    temp_top_users_rating.columns = ["sum_similarityIndex", "sum_weightedRating"]

    # Filter out entries where the similarity index is zero
    temp_top_users_rating = temp_top_users_rating[~(temp_top_users_rating["sum_similarityIndex"] == 0)]

    recommendation_df = pd.DataFrame()

    # Now we take the weighted average
    recommendation_df['rating'] = temp_top_users_rating['sum_weightedRating'] / temp_top_users_rating[
        'sum_similarityIndex']

    recommendation_df['movieId'] = temp_top_users_rating.index

    # recommendation_df['rating'] = recommendation_df['rating'].clip(1, 5)

    return recommendation_df


def task(arg):
    """
    Execute the KNN.
    """
    df, user_id, user = arg
    ratings = get_ratings(df, user_id, user)
    return user_id, ratings


def main():
    print("Reading data...")
    df = import_data('../../input/data_train.csv')
    df_pred = import_data('../../input/sampleSubmission.csv')
    df_pred["rating"] = df["rating"].mean()
    print(df["rating"].mean())
    print("Running threads...")
    predictions = {}
    try:
        # Execute code in parallel using multiprocessing.pool
        p = Pool()
        with p:
            for user_id, ratings in p.imap(task, get_input_user(df)):
                predictions[user_id] = ratings

    finally:
        # If we abort the execution or in case of an exception still save the currently calculated predictions
        # to file for further inspection
        print("Saving data...")
        export_predictions(np.c_[df_pred.userId.to_numpy(), df_pred.movieId.to_numpy(), df_pred.rating.to_numpy()],
                           predictions, "KNN")


if __name__ == "__main__":
    main()
