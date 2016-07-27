import pandas as pd
from scipy import sparse
import numpy as np
import json
from pprint import pprint
import gzip
from collections import defaultdict
import cPickle as pickle

def data_cleaner_csv(filename):
    '''
    INPUT: csv dataset
    OUTPUT: pandas dataframe

    Returns a cleaned dataset.
    '''
    # read data
    df = pd.read_csv(filename, header = None)
    # rename columns
    df.columns = ['userID', 'productID', 'rating', 'ratetime']
    # droping ratetime column
    df.drop('ratetime', axis=1, inplace=True)

    return df

def data_cleaner_json(path1, path2):
    '''
    INPUT: json.gzip datasets
    OUTPUT: pandas dataframes

    Returns two cleaned dataframes.
    '''

    df_movie_json = getDF(path1)
    df_game_json = getDF(path2)
    # drop few columns from movies and games dataframes
    cols_movie = ['description', 'related', 'price', 'price', 'salesRank', 'categories', 'brand']
    df_movie_json.drop(cols_movie, axis=1, inplace=True)
    cols_game = ['description', 'related', 'price', 'salesRank',  'brand']
    df_game_json.drop(cols_game, axis=1, inplace=True)
    # renaming columns
    df_movie_json.columns = [ 'productID', 'title' ,'imUrl']
    df_game_json.columns = [ 'productID','imUrl', 'categories', 'title']
    return df_movie_json, df_game_json

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def data_featuring_movies(df_movies_json, df_movies_csv):
    '''
    INPUT: pandas dataframes
    OUTPUT: pandas dataframes

    Returns a merged dataframe.
    '''
    # drop movies with no title
    df_movies_json.dropna(inplace=True, subset = ['title'])
    # merging two movies dataset
    return df_movies_csv.merge(df_movies_json, left_on='productID', right_on='productID', how='inner')

def data_featuring_games(df_games_json, df_games_csv):
    '''
    INPUT: pandas dataframes
    OUTPUT: pandas dataframes

    Returns a merged dataframe.
    '''
    # extracting info form descrption column
    cat_counts = defaultdict(int)
    df_games_json['test'] = df_games_json.categories.apply(lambda x: ''.join(x[0]))
    # selection only the rows that actually have video games
    df1 = df_games_json[df_games_json.test == 'Video GamesPCGames']
    df2 = df_games_json[df_games_json.test == 'Video GamesKids & Family']
    df3 = df_games_json[df_games_json.test == 'Video GamesDigital GamesCasual Games']
    df4 = df_games_json[df_games_json.test == 'Video GamesDigital GamesPC Game Downloads']
    df5 = df_games_json[df_games_json.test == 'Video GamesKids & FamilyNintendo DSGames']
    df6 = df_games_json[df_games_json.test == 'Video GamesKids & FamilyWiiGames']
    df7 = df_games_json[df_games_json.test == 'Video GamesXbox 360Games']
    df8 = df_games_json[df_games_json.test == 'Video GamesPlayStation 3Games']
    df9 = df_games_json[df_games_json.test == 'Video GamesMore SystemsPlayStation 2']
    df10 = df_games_json[df_games_json.test == 'Video GamesMore SystemsPlayStationGames']
    df11 = df_games_json[df_games_json.test == 'Video GamesKids & FamilyPlayStation 3Games']
    df12 = df_games_json[df_games_json.test == 'Video GamesKids & FamilyXbox 360Games']
    df13 = df_games_json[df_games_json.test == 'Video GamesWiiGames']
    df14 = df_games_json[df_games_json.test == 'Video Games']
    df15 = df_games_json[df_games_json.test == 'Video GamesPC']
    df16 = df_games_json[df_games_json.test == 'Video GamesSony PSPGames']
    # appending the rows selected
    df1 = df1.append(df2)
    df1 = df1.append(df3)
    df1 = df1.append(df4)
    df1 = df1.append(df5)
    df1 = df1.append(df6)
    df1 = df1.append(df7)
    df1 = df1.append(df8)
    df1 = df1.append(df9)
    df1 = df1.append(df10)
    df1 = df1.append(df11)
    df1 = df1.append(df12)
    df1 = df1.append(df13)
    df1 = df1.append(df14)
    df1 = df1.append(df15)
    df1 = df1.append(df16)
    # drop more columns
    cols = ['categories', 'title', 'test']
    df1.drop(cols, axis=1, inplace=True)
    # merge games dataframe
    #df_no_dupl = df_games_csv.drop_duplicates(subset=['productID'])
    return df_games_csv.merge(df1, left_on='productID', right_on='productID', how='inner')

def data_intersection(df1,df2):
    movie_users = set(df1['userID'].tolist())
    game_users = set(df2['userID'].tolist())
    intersection_users = (movie_users & game_users)

    movie_mask = df1['userID'].isin(intersection_users)
    df_movie_ratings_intersection = df1[movie_mask].reset_index(drop=True)

    #saving input for movies in intersection (API autocomplete input)
    movies_from_inter(df_movie_ratings_intersection)

    game_mask = df2['userID'].isin(intersection_users)
    df_game_ratings_intersection = df2[game_mask].reset_index(drop=True)
    #saving input for games in intersection (API games sample input)
    games_from_inter(df_game_ratings_intersection)

    df_iter = pd.concat([df_movie_ratings_intersection, df_game_ratings_intersection ])

    # API output
    output_api(df_iter)
    return df_iter

def movies_from_inter(dict_movies):
    movies_input = {}
    for v1, v2 in zip(dict_movies['title'], dict_movies['productID']):
            movies_input[v1] = v2
    # saving to a pickle file
    pickle.dump( movies_input, open( "../data/movies_input.p", "wb" ) )

def games_from_inter(dict_games):
    games_input = zip(dict_games['imUrl'], dict_games['productID'])
    pickle.dump( games_input, open( "../data/games_input.p", "wb" ) )


def output_api(df):
    #saving output for movies and games
    mg_out = defaultdict(list)

    # converting df columns to lists
    productID_lst = df.productID.tolist()
    imUrl_lst = df.imUrl.tolist()

    # lists will be the values of the dictionary
    mg_out['productID'] = productID_lst
    mg_out['imUrl'] = imUrl_lst

    for v1, v2 in zip(mg_out['productID'], mg_out['imUrl']):
            if v1 not in mg_out:
                mg_out[v1].append(v2)
                mg_out[v1].append('http://amazon.com/dp/' + v1)

    # saving to a pickle file
    pickle.dump( mg_out, open( "../data/output_api.p", "wb" ) )


def api_endpoints_datasets(df1,df2):
    df1 = df1.drop_duplicates(subset=['userID'])
    df = df1.merge(df2.drop_duplicates(subset=['userID']), left_on='userID', right_on='userID', how='inner')
    df.dropna(inplace=True, subset = ['productID_x', 'productID_y'])

    # movies and games dictionaries
    dict_movies = defaultdict(list)
    dict_games = defaultdict(list)

    # converting df columns to lists
    productID_x_lst = df.productID_x.tolist()
    imUrl_x_lst = df.imUrl_x.tolist()
    title_lst = df.title.tolist()

    # lists will be the values of the dictionary
    dict_movies['productID'] = productID_x_lst
    dict_movies['imUrl'] = imUrl_x_lst
    dict_movies['title'] = title_lst

    #saving input for movies in intersection (API autocomplete input)
    movies_from_inter(dict_movies)

    # Adding games (from the intersection)
    productID_y_lst = df.productID_y.tolist()
    imUrl_y_lst = df.imUrl_y.tolist()
    # lists will be the values of the dictionary
    dict_games['productID'] = productID_y_lst
    dict_games['imUrl'] = imUrl_y_lst

    #saving input for games in intersection (API games sample input)
    games_from_inter(dict_games)

    #adding the two dictionaries for API output
    output_api(dict_movies, dict_games)


def df_inter_dict(df):
    #NEEDS TO BE FIXED
    # movies dictionary
    dict_movies = defaultdict(list)
    mg_inter = defaultdict(list)

    # converting df columns to lists
    users_lst = df.userID.tolist()
    productID_x_lst = df.productID_x.tolist()
    rating_x_lst = df.rating_x.tolist()
    imUrl_x_lst = df.imUrl_x.tolist()
    title_lst = df.title.tolist()
    # lists will be the values of the dictionary
    dict_movies['userID'] = users_lst
    dict_movies['productID'] = productID_x_lst
    dict_movies['rating'] = rating_x_lst
    dict_movies['imUrl'] = imUrl_x_lst
    dict_movies['title'] = title_lst

    # games dictionary
    dict_games = defaultdict(list)
    # converting df columns to lists
    users_lst2 = df.userID.tolist()
    productID_y_lst = df.productID_y.tolist()
    rating_y_lst = df.rating_y.tolist()
    imUrl_y_lst = df.imUrl_y.tolist()
    # lists will be the values of the dictionary
    dict_games['userID'] = users_lst2
    dict_games['productID'] = productID_y_lst
    dict_games['rating'] = rating_y_lst
    dict_games['imUrl'] = imUrl_y_lst

    #adding the two dictionaries
    dict_movies['userID'].extend(dict_games['userID'])
    dict_movies['productID'].extend(dict_games['productID'])
    dict_movies['rating'].extend(dict_games['rating'])
    dict_movies['imUrl'].extend(dict_games['imUrl'])

    # Creating SFrame dictionary for the intersection
    mg_inter['userID'] = dict_movies['userID']
    mg_inter['productID'] = dict_movies['productID']
    mg_inter['rating'] = dict_movies['rating']

    df_inters = pd.DataFrame(mg_inter)

    # saving to a pickle file
    pickle.dump( df_inters, open( "../data/movies_games.p", "wb" ) )

    return  mg_inter

def top_games(df):
    # selecting games with rating 5
    df = df[df.rating == 5.0]
    # games dictionary
    dict_games = defaultdict(list)
    # converting df columns to lists
    users_lst = df.userID.tolist()
    productID_lst = df.productID.tolist()
    rating_lst = df.rating.tolist()
    imUrl_lst = df.imUrl.tolist()
    # lists will be the values of the dictionary
    dict_games['userID'] = users_lst
    dict_games['productID'] = productID_lst
    dict_games['rating'] = rating_lst
    dict_games['imUrl'] = imUrl_lst

    # saving to a pickle file
    pickle.dump( dict_games, open( "../data/best_games.p", "wb" ) )

def save_games(df):
    # games dictionary
    dict_games = defaultdict(list)
    # converting df columns to lists
    users_lst = df.userID.tolist()
    productID_lst = df.productID.tolist()
    rating_lst = df.rating.tolist()
    imUrl_lst = df.imUrl.tolist()
    # lists will be the values of the dictionary
    dict_games['userID'] = users_lst
    dict_games['productID'] = productID_lst
    dict_games['rating'] = rating_lst
    dict_games['imUrl'] = imUrl_lst

    # saving to a pickle file
    pickle.dump( dict_games, open( "../data/games.p", "wb" ) )


def save_movies(df):
    # movies dictionary
    dict_movies = defaultdict(list)
    # converting df columns to lists
    users_lst = df.userID.tolist()
    productID_lst = df.productID.tolist()
    rating_lst = df.rating.tolist()
    imUrl_lst = df.imUrl.tolist()
    title_lst = df.title.tolist()
    # lists will be the values of the dictionary
    dict_movies['userID'] = users_lst
    dict_movies['productID'] = productID_lst
    dict_movies['rating'] = rating_lst
    dict_movies['imUrl'] = imUrl_lst
    dict_movies['title'] = title_lst

    # saving to a pickle file
    pickle.dump( dict_movies, open( "../data/movies.p", "wb" ) )


if __name__ == '__main__':
    # Getting the the AP1 endpoints datasets
    # cleaning csv datasets
    # csv data files
    movies_csv = '../data/ratings_Movies_and_TV.csv'
    games_csv = '../data/ratings_Video_Games.csv'

    # json.gz data files
    movies_json = '../data/meta_Movies_and_TV.json.gz'
    games_json = '../data/meta_Video_Games.json.gz'


    df_movies_csv = data_cleaner_csv(movies_csv)
    df_games_csv = data_cleaner_csv(games_csv)

    # cleaning json.gz data files
    df_movies_json, df_games_json = data_cleaner_json(movies_json, games_json)

    # data featuring for the movies
    df_movies = data_featuring_movies(df_movies_json, df_movies_csv)
    # data featuring for the games
    df_games = data_featuring_games(df_games_json, df_games_csv)

    # creating input/output
    api_endpoints_datasets(df_movies, df_games)

    # other datasets
    # saving best games
    top_games(df_games)
    # saving games and movies
    save_games(df_games)
    save_movies(df_movies)
