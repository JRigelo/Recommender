""" Recommender capstone model: latent factors, item-item, user-user
"""
import pandas as pd
from scipy import sparse
import numpy as np
import data_prep as dp
import graphlab as gl
import json
from pprint import pprint
import training_model as tm
import cPickle as pickle
import random

# load games and movies dictionary
games_data = pickle.load( open( "../../data/games.p", "rb" ) )
# load best games data
best_games = pickle.load( open( "../../data/best_games.p", "rb" ) )
# load intersection data
df_inters = pickle.load( open( "../../data/movies_games.p", "rb" ) )

def load_sparse_matrix(filename):
    y = np.load(filename)
    z = sparse.coo_matrix((y['data'],(y['row'],y['col'])),shape=y['shape'])
    return z.tolil()

def load_input_data(filename):
    return gl.load_model(filename)

def games_movies_recom(user, item_ids, item_ratings, train_model):
    """INPUT: a string, two lists, trained model.
       OUPUT: a list of recommended products"""
    users_ids = []
    for i in range(0,len(item_ids)):
        users_ids.append(user)

    # making a suggestion model
    newdata = gl.SFrame({'userID': users_ids, 'productID': item_ids, 'rating': item_ratings})
    return train_model.recommend(users=[user], k=3, new_observation_data=newdata)

def norm(ratings_dict):
    n = 0.0
    for item, rating in ratings_dict.iteritems():
        n += rating*rating
    return np.sqrt(n)

def calc_similarity(old_ratings_dict, new_ratings_dict):
    score = 0.0
    for item, rating in old_ratings_dict.iteritems():
        if item in new_ratings_dict:
            score += rating * new_ratings_dict[item]
    return score / (norm(old_ratings_dict) * norm(new_ratings_dict))

def build_user_group_dictionary(df):
    groups = df.groupby('userID')
    dict_ = groups.apply(lambda innerdf: innerdf.to_dict(orient='list')).to_dict()
    def convert(value):
        return {itemid: rating for itemid, rating in zip(value['productID'], value['rating'])}
    return {userid: convert(value) for userid, value in dict_.iteritems()}

def find_similar_user(user_groups, item_ids, ratings):
    new_ratings_dict = {item: rating for item, rating in zip(item_ids, ratings)}
    sims = [(user, calc_similarity(old_ratings_dict, new_ratings_dict))
                for user, old_ratings_dict in user_groups.iteritems()]
    sims = sorted(sims, cmp=lambda x,y: -cmp(x[1], y[1]))
    return sims[0][0]


def recommender_process(user, item_ids, item_ratings, train_model_mg, train_model_m, train_model_g, data_folder):
    # getting the best recommendations from movies & games model
    #best_recom_mg = games_movies_recom(user, item_ids, item_ratings, train_model_mg)
    users_ids = []
    for i in range(0,len(item_ids)):
        users_ids.append(user)

    intersection_user_group_dict = build_user_group_dictionary(df_inters)
    sim_user = find_similar_user(intersection_user_group_dict, item_ids,
        item_ratings)

    best_recom_mg = train_model_mg.recommend(users=[sim_user], k=3)


    """# making a suggestion model
    newdata = gl.SFrame({'userID': users_ids, 'productID': item_ids, 'rating': item_ratings})
    best_recom_mg = train_model_mg.recommend(users=[user], k=3, new_observation_data=newdata)
    """

    print 'best_recom_mg productID', best_recom_mg['productID']

    # Similar items within other datasets
    recm_sim_mg = train_model_m.get_similar_items(best_recom_mg['productID'], k=1)
    recg_sim_mg = train_model_g.get_similar_items(best_recom_mg['productID'], k=1)

    # Putting the recommendations together
    recom = final_recom_(recm_sim_mg, recg_sim_mg)

    # I there are no games suggested we can get some top rated one randomly
    # Maybe change to popular?
    recom = if_no_game(recom, games_data, data_folder)

    return recom

def final_recom_(recm_sim_mg, recg_sim_mg):
    final = []
    for item in recm_sim_mg['productID']:
        if item not in final:
            final.append(item)
    for item in recg_sim_mg['productID']:
        if item not in final:
            final.append(item)

    final.extend(recm_sim_mg['similar'])
    final.extend(recg_sim_mg['similar'])
    return final

def if_no_game(final_recommendation, games_data, data_folder):
    # load best games data
    #best_games = pickle.load( open( data_folder + "/best_games.p", "rb" ) )

    diff = list(set(games_data['productID']) - set(final_recommendation))

    if len(diff) == len(games_data['productID']):
        # getting some random best games
        final_recommendation.extend(random.sample(best_games['productID'], 3))
    return final_recommendation


if __name__ =='__main__':
    # Loading trainning model
    t_model_m_g = load_input_data('../data/model_m_g')
    t_model_m = load_input_data('../data/model_m')
    t_model_g = load_input_data('../data/model_g')


    # toy example to test recommender
    user = 'Joyce'
    item_ids = ['0439671418', 'B00004W0W7', '0439715571', '6301759338', '0700099867']

    #            '3868832815', '0970154097', 'B00007JME6', '1886846847', 'B0007MWZIG','B0001VL0K2']
    item_ratings = [1, 1, 5, 3, 1] #1, 4, 3, 1, 1, 4]

    # getting the recommendations from the three models
    recom = recommender_process(user, item_ids, item_ratings,
                                t_model_m_g, t_model_m, t_model_g,
                                '../data')
    print 'Recommended movies from mg', recom
    '''print 'Recommended games from mg', recg_sim_mg
    print ''

    print 'Recommended users(movies) from mg', userm_sim_mg
    print 'Recommended users(games) from mg', userg_sim_mg
    print '''''

    """print 'Recommended movies/games from m', recmg_sim_m
    print 'Recommended games from m', recg_sim_m
    print ''

    print 'Recommended movies/games from g', recmg_sim_m
    print 'Recommended movies from g', recg_sim_m
    print ''
    """
    """
    load games and movies dictionary
    games_data = pickle.load( open( "../data/games.p", "rb" ) )
    movies_data = pickle.load( open( "../data/movies.p", "rb" ) )

    recom = final_recom_(recm_sim_mg, recg_sim_mg)
    print "recommendation", recom
    recom = if_no_game(recom, games_data)

    print "final recommendation", recom
    """
