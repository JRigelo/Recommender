""" Recommender capstone model: SVD, NMF, item-item, user-user

"""
import pandas as pd
from scipy import sparse
import numpy as np
import data_prep as dp
import graphlab as gl
import json
from pprint import pprint

def load_sparse_matrix(filename):
    y = np.load(filename)
    z = sparse.coo_matrix((y['data'],(y['row'],y['col'])),shape=y['shape'])
    return z.tolil()

def load_input_data(filename):
    return gl.load_model(filename)

def recom_products(user, item_ids, item_ratings, train_model):
    """INPUT: a string, two lists, trained model.
       OUPUT: a list of recommended products"""
    users_ids = []
    for i in range(0,len(item_ids)):
        users_ids.append(user)

    # making a suggestion model
    newdata = gl.SFrame({'userID': users_ids, 'productID': item_ids, 'rating': item_ratings})
    return train_model.recommend(users=[user], new_observation_data=newdata)

if __name__ =='__main__':
    # Loading trainning model
    t_model_m_g = load_input_data('../data/model_m_g')
    t_model_m = load_input_data('../data/model_m')
    t_model_g = load_input_data('../data/model_g')

    # toy example to test recommender
    user = 'Joyce'
    item_ids = ['0439671418', 'B00004W0W7', '0439715571', '6301759338', '0700099867', \
                '3868832815', '0970154097', 'B00007JME6', '1886846847', 'B0007MWZIG']
    item_ratings = [1, 1, 5, 3, 1, 1, 4, 3, 1, 1]

    # making a suggestion m_g model
    recommendation_m_g = recom_products(user, item_ids, item_ratings, t_model_m_g)
    # making a suggestion m model
    recommendation_m = recom_products(user, item_ids, item_ratings, t_model_m)
    # making a suggestion g model
    recommendation_g = recom_products(user, item_ids, item_ratings, t_model_g)
