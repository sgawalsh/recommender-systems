# -*- coding: utf-8 -*-

import pickle
import numpy as np
from scipy.spatial import cKDTree
import data
import sgd_mf
import nn

def load_mf_embeds():
    user_vec = pickle.load(open("mf_emeddings/user_matrix", "rb" ))
    item_vec = pickle.load(open("mf_emeddings/item_matrix", "rb" ))
    
    return user_vec, item_vec

def load_nn_embeds():
    users = pickle.load(open("models/user_embeds", "rb" ))
    items = pickle.load(open("models/item_embeds", "rb" ))
    
    return users.cpu().weight.detach().numpy(), np.array(items.cpu().weight.detach().numpy())

def get_n_closest(in_matrix, i, n = 5):
    search_vec = in_matrix[i]
    search_matrix = np.delete(in_matrix, i, axis = 0)
    ret_matrix = cKDTree(search_matrix).query(search_vec, k = n)
    ret_matrix[1][ret_matrix[1] >= i] += 1
    return ret_matrix
    

def content_based():
    user_vec, item_vec = load_nn_embeds()
    # user_vec, item_vec = load_mf_embeds()
    
    for i in range(10):
        n_closest = get_n_closest(item_vec, i)
        movies = data.get_data_raw("ml-latest-small/movies.csv").to_numpy()
        recommendations = movies[n_closest[1]]
        print("For: {}\n".format(movies[i][1]))
        print("{}\n".format(recommendations))

def compare_preds(u_id = 0, n_compare = 10, mf_index = True):
    mf_preds = sgd_mf.load_predictions(u_id, 0, False)
    nn_preds = nn.get_predictions(u_id, 0, False)
    
    index = mf_preds[:,0] if mf_index else nn_preds[:,0] # set index and query prediction matrices
    query = nn_preds[:,0] if mf_index else mf_preds[:,0]
        
    sum = 0
    for i in range(n_compare):
        m_id = index[i]
        q_index = np.where(query == m_id)[0][0]
        #print("Index # {} is Query # {}".format(i, q_index))
        sum += q_index
    
    print("Average placement is {} / {}".format(sum / n_compare, index.__len__()))
        
for i in range(10):
    compare_preds(i, mf_index = False)