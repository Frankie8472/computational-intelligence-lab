from porter import export_data, import_dataframe
from math import sqrt
import pandas as pd
import numpy as np
from multiprocessing import Pool

# Implementation after: https://medium.com/swlh/how-to-build-simple-recommender-systems-in-python-647e5bcd78bd

def get_input_user(data):
    for i in range(len(data['userId'].unique())):
        yield data, i, data[data.userId == i]
        
def get_ratings(df, user_id, curr_user):
    print("Using userId {}".format(user_id))
    
    userSubset = df[(df['movieId'].isin(curr_user['movieId'].tolist())) & (df.userId != user_id)]
    userSubsetGroup = userSubset.groupby(['userId'])
    sizes = userSubset.groupby(['userId']).userId.agg('count').to_frame('c').reset_index()
    sizes = sizes.sort_values(by="c", ascending=False)[0:100]
    userSubset = userSubset[userSubset['userId'].isin(sizes["userId"].tolist())]
    userSubsetGroup = userSubset.groupby(['userId'])
    
    print("Calculating similarity for user {}...".format(user_id))
    # Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
    
    pearsonCorrelationDict = {}
    
    # For every user group in our subset
    for name, group in userSubsetGroup:
        # Let's start by sorting the input and current user group so the values aren't mixed up later on
        group = group.sort_values(by='movieId')
        inputMovies = curr_user.sort_values(by='movieId')
        # Get the N for the formula
        nRatings = len(group)
        # Get the review scores for the movies that they both have in common
        temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
        # And then store them in a temporary buffer variable in a list format to facilitate future calculations
        tempRatingList = temp_df['rating'].tolist()
        # Let's also put the current user group reviews in a list format
        tempGroupList = group['rating'].tolist()
        # Now let's calculate the pearson correlation between two users, so called, x and y
        Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
        Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
        Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
        # If the denominator is different than zero, then divide, else, 0 correlation.
        if Sxx != 0 and Syy != 0:
            pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
        else:
            pearsonCorrelationDict[name] = 0
    pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
    pearsonDF.columns = ['similarityIndex']
    pearsonDF['userId'] = pearsonDF.index
    pearsonDF.index = range(len(pearsonDF))
    topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
    
    print("Calculating rating for user {}...".format(user_id))
    
    topUsersRating=topUsers.merge(df, left_on='userId', right_on='userId', how='inner')
    
    # Multiplies the similarity by the user's ratings
    topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
    
    # Applies a sum to the topUsers after grouping it up by userId
    
    tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
    
    tempTopUsersRating.columns = ['sum_similarityIndex', 'sum_weightedRating']
    
    # Creates an empty dataframe
    
    recommendation_df = pd.DataFrame()
    
    # Now we take the weighted average
    
    recommendation_df['rating'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
    
    recommendation_df['movieId'] = tempTopUsersRating.index
    
    # recommendation_df = recommendation_df.sort_values(by='rating', ascending=False)
    
    return recommendation_df

def task(arg):
    df, user_id, user = arg
    ratings = get_ratings(df, user_id, user)
    return user_id, ratings
    
    
def write_to_matrix(df_pred, user_id, ratings):
    print("Writing to matrix for user {}...".format(user_id))
    for i in df_pred[df_pred.userId == user_id]["movieId"]:
        if i in ratings["movieId"]:
            df_pred.loc[(df_pred.userId == user_id) & (df_pred.movieId == i),'rating'] = ratings.at[i, "rating"]

def export_predictions(base, predictions, name):
    print(len(base))
    print(len(base[:]))
    no_ratings_count = 0
    with open('../output/' + name + 'Submission.csv', 'w') as file:
        file.write('Id,Prediction')
        for i in range(0, len(base)):
            row = int(base[i][0] + 1)
            col = int(base[i][1] + 1)
            val = 3
            df = predictions.get(row-1)
            if df is not None:
                try:
                    val = df.at[col - 1, 'rating']
                except KeyError:
                    print("*WARNING*: No rating for user {} for movie {}".format(row - 1, col - 1))
                    no_ratings_count+=1
            if not (1 <= val <= 5):
                print("WARNING: user {} for movie {} has invalid rating {}. Clamping.".format(row - 1, col - 1, val))
                val = min(5, max(1, val))
            file.write("\n")
            file.write('r' + str(row) + '_c' + str(col) + ',' + str(val))
    if no_ratings_count:
        print("Had {} occurencies that could not be rated".format(no_ratings_count))

def main():
    print("Reading data...")
    df = import_dataframe()
    df_pred = import_dataframe(filepath='../output/sampleSubmission.csv')
    nr_of_users = len(df['userId'].unique())
    nr_of_movies = len(df['movieId'].unique())
    print("Running threads...")
    predictions = {}
    try:
        p = Pool()
        with p:
            for user_id, ratings in p.imap(task, get_input_user(df), 1):
                #write_to_matrix(df_pred, user_id, ratings)
                predictions[user_id] = ratings[ratings['movieId'].isin(df_pred[df_pred.userId == user_id]['movieId'].tolist())]
        p.join()
    finally:
        print("Saving data...")
        export_predictions(np.c_[df_pred.userId.to_numpy(), df_pred.movieId.to_numpy()], predictions, "josua")

if __name__ == "__main__":
    main()