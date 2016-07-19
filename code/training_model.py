""" Training model: SVD, NMF, item-item, user-user
"""
import pandas as pd
from scipy import sparse
import numpy as np
import data_prep as dp
import graphlab as gl
import json
from pprint import pprint


def data_prep_(path1, path2, path3, path4):
    # cleaning csv datasets
    df_movies_csv = dp.data_cleaner_csv(path1)
    df_games_csv = dp.data_cleaner_csv(path2)

    # cleaning json.gz data files
    df_movies_json, df_games_json = \
        dp.data_cleaner_json(path3, path4)

    # data featuring for the movies
    df_movies = dp.data_featuring_movies(df_movies_json, df_movies_csv)
    # data featuring for the games
    df_games = dp.data_featuring_games(df_games_json, df_games_csv)

    # inner join of two datasets on usersID
    df_inter = dp.data_intersection(df_movies, df_games)
    # converting inner joined dataframe to a dictionary
    movies_and_games = dp.df_inter_dict(df_inter)

    return movies_and_games, df_movies, df_games

def training_data(train_data, dataframe = True):
    """Create a FactorizationRecommender
    that learns latent factors for each user and item and uses
    them to make rating predictions."""

    if dataframe:
        # SFrame (dictionary from df)
        sf = gl.SFrame(train_data[['userID', 'productID', 'rating']])
    else:
        sf = gl.SFrame(train_data)

    # Create a matrix factorization model
    rec = gl.recommender.factorization_recommender.create(
            sf,
            user_id='userID',
            item_id='productID',
            target='rating',
            #solver='als',
            max_iterations=1000,
            side_data_factorization=False)
    return rec

def save_model(rec, path):
    rec.save(path)

if __name__ =='__main__':
    # For Graphlab usage we need to use dataframes or dictionaries as input
    # csv data files
    movies_csv = '../data/ratings_Movies_and_TV.csv'
    games_csv = '../data/ratings_Video_Games.csv'

    # json.gz data files
    movies_json = '../data/meta_Movies_and_TV.json.gz'
    games_json = '../data/meta_Video_Games.json.gz'

    # prepping the datasets data
    movies_and_games, df_movies, df_games = \
        data_prep_(movies_csv, games_csv, movies_json, games_json)

    # training Data
    train_m_g = training_data(movies_and_games, dataframe = False)
    train_m = training_data(df_movies)
    train_g = training_data(df_games)

    # saving models
    save_model(train_m_g, '../data/model_m_g')
    save_model(train_m, '../data/model_m')
    save_model(train_g, '../data/model_g')
