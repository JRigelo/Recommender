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
    #df_no_dupl = df_movies_csv.drop_duplicates(subset=['productID'])
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
    df1_no_dupl = df1.drop_duplicates(subset=['userID'])
    return df1_no_dupl.merge(df2.drop_duplicates(subset=['userID']), left_on='userID',\
            right_on='userID', how='inner')

def df_inter_dict(df):
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

    #saving input for movies in intersection
    """movies_input = {}
    for v1, v2 in zip(dict_movies['title'], dict_movies['productID']):
            movies_input[v1] = v2
    # saving to a pickle file
    pickle.dump( movies_input, open( "../data/movies_input.p", "wb" ) )
    """
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
    """
    #saving input for games in intersection
    games_input = zip(dict_games['imUrl'], dict_games['productID'])
    pickle.dump( games_input, open( "../data/games_input.p", "wb" ) )
    """
    #adding the two dictionaries
    dict_movies['userID'].extend(dict_games['userID'])
    dict_movies['productID'].extend(dict_games['productID'])
    dict_movies['rating'].extend(dict_games['rating'])
    dict_movies['imUrl'].extend(dict_games['imUrl'])

    # Creating SFrame dictionary for the intersection
    mg_inter['userID'] = dict_movies['userID']
    mg_inter['productID'] = dict_movies['productID']
    mg_inter['rating'] = dict_movies['rating']

    return mg_inter

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

    return dict_games

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

    return dict_movies


def subtract_intersection(df1,df_inner):
    users_lst = df_inner.userID.tolist()
    df1_no_dupl = df1.drop_duplicates(subset=['userID'])
    return  df1_no_dupl.query('userID not in @users_lst')


def utility_matrix_lil(df):
    '''
    INPUT: pandas dataframe
    OUTPUT: scipy sparse matrix

    Returns two dictionaries for user and product IDs and an utility matrix
    '''
    user_unique = df.userID.unique()
    product_unique = df.productID.unique()
    user_num_ID = {name: index for index, name in enumerate(user_unique)}
    product_num_ID = {name: index for index, name in enumerate(product_unique)}
    util_matrix = sparse.lil_matrix((len(user_unique), len(product_unique)))
    for _, row in df.iterrows():
        util_matrix[user_num_ID[row.userID], product_num_ID[row.productID]] = row.rating
    return user_num_ID, product_num_ID, util_matrix


def matrix_intersect(df_inner):
    '''
    INPUT: pandas dataframe
    OUTPUT: scipy sparse matrix

    Returns three dictionaries for user and product IDs and an utility matrix
    '''
    user_unique = df_inner.userID.unique()
    product_unique_x = df_inner.productID_x.unique()
    product_unique_y = df_inner.productID_y.unique()
    user_num_ID = {name: index for index, name in enumerate(user_unique)}
    product_ID_x = {name: index for index, name in enumerate(product_unique_x)}
    product_ID_y = {name: index for index, name in enumerate(product_unique_y)}

    util_matrix_x = sparse.lil_matrix((len(user_unique), len(product_unique_x)))
    util_matrix_y = sparse.lil_matrix((len(user_unique), len(product_unique_y)))
    for _, row in df_inner.iterrows():
        util_matrix_x[user_num_ID[row.userID], product_ID_x[row.productID_x]] = row.rating_x
    for _, row in df_inner.iterrows():
        util_matrix_y[user_num_ID[row.userID], product_ID_y[row.productID_y]] = row.rating_y
    # concatenating two matrices by column
    util_matrix = sparse.hstack((util_matrix_x, util_matrix_y))
    return user_num_ID, product_ID_x, product_ID_y, util_matrix


def block_matrix(matrix1, matrix2):
    '''
    INPUT: two scipy sparse matrices
    OUTPUT: one scipy sparse matrix

    Returns a concatenated utility matrix
    '''
    return sparse.block_diag((matrix1, matrix2)).toarray()

def concat(final_matrix, inner_matrix):
    '''
    INPUT: two scipy sparse matrices
    OUTPUT: one scipy sparse matrix

    Returns a concatenated utility matrix
    '''
    return sparse.vstack((inner_matrix, final_matrix))

def save_sparse_matrix(filename,x):
    x_coo = x.tocoo()
    row = x_coo.row
    col = x_coo.col
    data = x_coo.data
    shape = x_coo.shape
    np.savez(filename,row=row,col=col,data=data,shape=shape)

def output_api(dict_movies, dict_games):
    #saving output for movies and games
    mg_out = defaultdict(list)
    for v1, v2 in zip(dict_movies['productID'], dict_movies['imUrl']):
            mg_out[v1].append(v2)
            mg_out[v1].append('http://amazon.com/dp/' + v1)

    for v1, v2 in zip(dict_games['productID'], dict_games['imUrl']):
            mg_out[v1].append(v2)
            mg_out[v1].append('http://amazon.com/dp/' + v1)
    # saving to a pickle file
    pickle.dump( mg_out, open( "../data/output_api.p", "wb" ) )

    return mg_out


if __name__ == '__main__':
    # csv data files
    games_file = '../data/ratings_Video_Games.csv'
    movies_file = '../data/ratings_Movies_and_TV.csv'

    # cleaning csv datasets
    df_games = data_cleaner_csv(games_file)
    df_movies = data_cleaner_csv(movies_file)


    # inner join of two datasets on usersID
    df_inter = data_intersection(df_movies, df_games)

    # subtracting common users
    df_games = subtract_intersection(df_games, df_inter)
    df_movies = subtract_intersection(df_movies, df_inter)

    # creating utility matrices
    user_games_ID, prod_games_ID, util_games_M = utility_matrix_lil(df_games)
    user_movies_ID, prod_movies_ID, util_movies_M = utility_matrix_lil(df_movies)

    # Preping the intersection matrix
    user_games_ID, prod_movies_ID, prod_movies_ID, movies_game_M = matrix_intersect(df_inter)
