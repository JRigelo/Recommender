import pandas as pd
from scipy import sparse
import numpy as np

def data_cleaner(filename):
    '''
    INPUT: csv dataset
    OUTPUT: pandas dataframe

    Returns a cleaned dataset.
    '''
    # read data
    df = pd.read_csv(filename)
    # rename columns
    df.columns = ['userID', 'productID', 'rating', 'ratetime']
    # droping ratetime column
    df.drop('ratetime', axis=1, inplace=True)
    return df

def data_intersection(df1,df2):
    df1_no_dupl = df1.drop_duplicates(subset=['userID'])
    return df1_no_dupl.merge(df2.drop_duplicates(subset=['userID']), left_on='userID', right_on='userID', how='inner')


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

if __name__ == '__main__':
    # Data files
    games_file = '../data/ratings_Video_Games.csv'
    movies_file = '../data/ratings_Movies_and_TV.csv'

    # Cleaning datasets
    df_games = data_cleaner(games_file)
    df_movies = data_cleaner(movies_file)

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

    # Save matrices
#    save_sparse_matrix("../data/movies_games_M", movies_game_M)
#    save_sparse_matrix("../data/movies_M", util_movies_M)
#    save_sparse_matrix("../data/games_M", util_games_M)


    print 'movies_game_M', movies_game_M
