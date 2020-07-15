from porter import export_data, import_dataframe
from math import sqrt
import pandas as pd

# Implementation after: https://medium.com/swlh/how-to-build-simple-recommender-systems-in-python-647e5bcd78bd

def get_input_user(data):
    for i in range(len(data)):
        yield data[data.userId == i], i


def main():
    df = import_dataframe()
    df_pred = import_dataframe(filepath='../output/sampleSubmission.csv')
    user = get_input_user(df)
    curr_user, user_id = next(user)
    
    userSubset = df[(df['movieId'].isin(curr_user['movieId'].tolist())) & (df.userId != user_id)]
    userSubsetGroup = userSubset.groupby(['userId'])
    userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)
    userSubsetGroup = userSubsetGroup[0:100]
    
    #Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
    
    pearsonCorrelationDict = {}
    
    # For every user group in our subset
    for name, group in userSubsetGroup:
        # Let's start by sorting the input and current user group so the values aren't mixed up later on
        group = group.sort_values(by='movieId')
        inputMovies = curr_user.sort_values(by='movieId')
        # Get the N for the formula
        nRatings = len(group)
        # Get the review scores for the movies that they both have in common
        temp_df = curr_user[curr_user['movieId'].isin(group['movieId'].tolist())]
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
    
    topUsersRating=topUsers.merge(df, left_on='userId', right_on='userId', how='inner')
    
    #Multiplies the similarity by the user's ratings
    topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
    
    #Applies a sum to the topUsers after grouping it up by userId
    
    tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
    
    tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
    
    #Creates an empty dataframe
    
    recommendation_df = pd.DataFrame()
    
    #Now we take the weighted average
    
    recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
    
    recommendation_df['movieId'] = tempTopUsersRating.index
    
    recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
    
    print(recommendation_df.head(10))
    
    
main()