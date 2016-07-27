""" Training model: matrix factorization

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


    return df_inter, df_movies, df_games


def training_data(train_data, dictionary = True):
    """the factorization_recommender will return the
       nearest items based on the cosine similarity between
        latent item factors."""
    #print 'train_data', train_data['userID']
    if dictionary:
        sf = gl.SFrame(train_data)
    else:
        sf = gl.SFrame(train_data[['userID', 'productID', 'rating']])
    #print 'sf', sf

    # Create a matrix factorization model
    rec = gl.recommender.factorization_recommender.create(
            sf,
            user_id='userID',
            item_id='productID',
            target='rating',
            linear_regularization=1e-09,
            max_iterations=50,
            num_factors=16,
            regularization= 1e-07,
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

    # prepping the datasets
    movies_and_games, df_movies, df_games = \
        data_prep_(movies_csv, games_csv, movies_json, games_json)

    # training Data
    train_m_g = training_data(movies_and_games, dictionary = False)
    train_m = training_data(df_movies, dictionary = False)
    train_g = training_data(df_games, dictionary = False)

    # saving models
    save_model(train_m_g, '../data/model_m_g')
    save_model(train_m, '../data/model_m')
    save_model(train_g, '../data/model_g')

    # other datasets for API
    # saving best games
    dp.top_games(df_games)

    # saving games and movies
    dp.save_games(df_games)
    dp.save_movies(df_movies)
