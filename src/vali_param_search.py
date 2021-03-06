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

def param_search_(train_data, dataframe=True, cv=5):
    if dataframe:
        # SFrame (dictionary from df)
        sf = gl.SFrame(train_data[['userID', 'productID', 'rating']])
    else:
        sf = gl.SFrame(train_data)

    kfolds = gl.cross_validation.KFold(sf, cv)
    params = dict(user_id='userID', item_id='productID', target='rating',
                  side_data_factorization=False)
    paramsearch = gl.model_parameter_search.create(
                        kfolds,
                        gl.recommender.ranking_factorization_recommender.create,
                        params)

    return paramsearch

def validation_(train_data, dataframe=True, cv=5):
    if dataframe:
        # SFrame (dictionary from df)
        sf = gl.SFrame(train_data[['userID', 'productID', 'rating']])
    else:
        sf = gl.SFrame(train_data)
    kfolds = gl.cross_validation.KFold(sf, cv)
    params = dict(user_id='userID', item_id='productID', target='rating',
                  linear_regularization=1e-09, max_iterations=25, num_factors=8,
                  regularization= 0.0001, side_data_factorization=False)
    job = gl.cross_validation.cross_val_score(kfolds,
                                              gl.ranking_factorization_recommender.create,
                                              params)
    return job.get_results()


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

    # parameter search
    param_search = param_search_(movies_and_games)
    #print 'get_status', param_search.get_status()
    #print 'get_metrics', param_search.get_metrics()
    #print 'get_results', param_search.get_results()

    print "best params by recall@5:"
    pprint(param_search.get_best_params('mean_validation_recall@5'))
    print

    print "best params by precision@5:"
    pprint(param_search.get_best_params('mean_validation_precision@5'))
    print

    print "best params by rmse:"
    pprint(param_search.get_best_params('mean_validation_rmse'))


    # cross validation
    #val_games = validation_(df_games)
    #print 'val_games', val_games
    #val_mg = validation_(movies_and_games)
    #print 'val_mg', val_mg
    #val_movies = validation_(df_movies)
    #print 'val_movies', val_movies
