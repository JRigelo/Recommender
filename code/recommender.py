""" Recommender capstone model: SVD, NMF, item-item, user-user

Data:
+------+-------+--------+
| user | movie | rating |
+------+-------+--------+
| 196  |  242  |   3    |
| 186  |  302  |   3    |
|  22  |  377  |   1    |
| 244  |   51  |   2    |
| 166  |  346  |   1    |
| 298  |  474  |   4    |
| 115  |  265  |   2    |
| 253  |  465  |   5    |
| 305  |  451  |   3    |
|  6   |   86  |   3    |
+------+-------+--------+

"""
import pandas as pd
from scipy import sparse
import numpy as np
import data_prep as dp
import graphlab as gl

def load_sparse_matrix(filename):
    y = np.load(filename)
    z = sparse.coo_matrix((y['data'],(y['row'],y['col'])),shape=y['shape'])
    return z.tolil()

def training_data(df):
    """Create a FactorizationRecommender
    that learns latent factors for each user and item and uses
    them to make rating predictions."""

    # SFrame (dictionary from df)
    sf = gl.SFrame(df[['userID', 'productID', 'rating']])

    # Create a matrix factorization model
    rec = gl.recommender.factorization_recommender.create(
            sf,
            user_id='userID',
            item_id='productID',
            target='rating',
            solver='als',
            side_data_factorization=False)
    rec.save('../data/model_test')
    return rec

def pred_products(user, item_ids, item_ratings, rec):
    """INPUT: three lists, trained matrix fact.
       OUPUT:
    """
    newdata = gl.SFrame({'userID':  user , 'productID': item_ids, 'rating': item_ratings})
    return rec.recommend(users=['Chris'], new_observation_data=newdata)





if __name__ =='__main__':
    games_M = load_sparse_matrix('../data/movies_games_M.npz')

    #For Graphlab usage we need to go back to use dataframes as input
    # Data files
    games_file = '../data/ratings_Video_Games.csv'
    movies_file = '../data/ratings_Movies_and_TV.csv'

    # Cleaning datasets
    df_games = dp.data_cleaner(games_file)
    df_movies = dp.data_cleaner(movies_file)

    # inner join of two datasets on usersID
    df_inter = dp.data_intersection(df_movies, df_games)

    #SFrame applied to the inner-joined dataframe
    sf_mg = gl.SFrame(df_inter[['userID', 'productID_x', 'rating_x', 'productID_y','rating_y']])

    # Training Data
    #train_m = training_data(df_movies)
    train_g = training_data(df_games)

    trained_model_g = gl.load_model('../data/model_test')

    # making a suggestion
    user = ['Chris']
    item_ids = ["0078764343"]
    ratings = [3]
    recommendation = pred_products(user, item_ids, ratings, trained_model_g)
    print recommendation

    """# Predicting a movie
    one_datapoint_sf = gl.SFrame({'userID': [2], 'productID': [2]})
    print "rating:", rec_m.predict(one_datapoint_sf)[0]"""
