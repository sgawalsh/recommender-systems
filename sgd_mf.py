# -*- coding: utf-8 -*-
import numpy as np
from math import floor, sqrt
import matplotlib.pyplot as plt
from itertools import product
import data
import pickle

class sgdMF():
    def __init__(self, ratings, n_factors = 100, user_vec_reg = 0, item_vec_reg = 0, user_bias_reg = 0, item_bias_reg = 0, lr = .01, training_split = .9, populate_bias = True, set_seed = True):
        self.train_vals = ratings[~np.isnan(ratings)]
        self.ratings = (ratings - self.train_vals.mean()) / self.train_vals.std() # scale and center data
        self.n_factors = n_factors
        self.n_users, self.n_items = ratings.shape
        self.set_seed = set_seed
        if self.set_seed:
            np.random.seed(0)
        self.user_matrix = np.random.normal(scale=1./self.n_factors, size=(self.n_users, self.n_factors))
        if self.set_seed:
            np.random.seed(0)
        self.item_matrix = np.random.normal(scale=1./self.n_factors, size=(self.n_items, self.n_factors))
        self.user_bias_reg, self.item_bias_reg, self.user_vec_reg, self.item_vec_reg = user_bias_reg, item_bias_reg, user_vec_reg, item_vec_reg
        self.lr = lr
        self.test_mae, self.test_rmse, self.train_mae, self.train_rmse = [], [], [], []
        self.pop_bias = populate_bias
        
        self.gen_datasets(training_split)
        self.bias_init(populate_bias)
        self.min_error = np.inf

    def gen_datasets(self, training_split):
        dataset = np.column_stack(np.nan_to_num(self.ratings).nonzero())
        
        if training_split >= 1:
            self.train_data = dataset
        else:
            if self.set_seed:
                np.random.seed(0)
            np.random.shuffle(dataset)
            self.train_data, self.test_data = np.split(dataset, [floor(len(dataset) * training_split)])

    def bias_init(self, populate = True):
        #self.global_bias = np.sum(self.train_data) / np.count_nonzero(~np.isnan(self.ratings)) #not needed for centered input data
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        
        if populate:
            for u, row in enumerate(self.ratings):
                self.user_bias[u] = row[~np.isnan(row)].mean()
            for i, col in enumerate(self.ratings.T):
                self.item_bias[i] = col[~np.isnan(col)].mean()
                
    def train(self, test_period = 10000):
        if self.set_seed:
            np.random.seed(0)
        np.random.shuffle(self.train_data)
        mae_sum, mse_sum, test_counter = 0, 0, 0
        #print("{0}/{1} - Average Error: {2}".format(0, len(self.train_data), self.test()))
        for i, coord in enumerate(self.train_data):
            target = self.ratings[coord[0]][coord[1]]
            prediction = self.predict(coord[0], coord[1])
            err = target - prediction
            test_counter += 1
            mae_sum += abs(err)
            mse_sum += err * err
            
            self.update_sgd(err, coord[0], coord[1])
        
        
            
            if test_counter >= test_period:
                train_mae = mae_sum / test_period
                train_rmse = sqrt(mse_sum / test_period)
                self.train_mae.append(train_mae)
                self.train_rmse.append(train_rmse)
                mae_sum, mse_sum, test_counter = 0, 0, 0
                test_mae, test_rmse = self.test()
                print("{0}/{1} - Train MAE: {2}, Train MSE: {3}, Test MAE: {4}, Test MSE: {5}".format(i + 1, len(self.train_data), round(train_mae, 3), round(train_rmse, 3), round(test_mae, 3), round(test_rmse, 3)))
                if test_rmse < self.min_error:
                    self.min_error = test_rmse
                    self.best_user_matrix = np.copy(self.user_matrix)
                    self.best_item_matrix = np.copy(self.item_matrix)
                    self.best_user_bias = np.copy(self.user_bias)
                    self.best_item_bias = np.copy(self.item_bias)
                    
            
    def update_sgd(self, err, u_id, i_id):
        self.user_bias[u_id] += self.lr * (err - self.user_bias_reg * self.user_bias[u_id])
        self.item_bias[i_id] += self.lr * (err - self.item_bias_reg * self.item_bias[i_id])
        
        self.user_matrix[u_id] += self.lr * (err * self.item_matrix[i_id] - self.user_vec_reg * self.user_matrix[u_id])
        self.item_matrix[i_id] += self.lr * (err * self.user_matrix[u_id] - self.item_vec_reg * self.item_matrix[i_id])
        
    def predict(self, u_id, i_id):
        prediction = self.user_bias[u_id] + self.item_bias[i_id]
        prediction += self.user_matrix[u_id].dot(self.item_matrix[i_id])
        return prediction
    
    def test(self):
        mae_sum, mse_sum = 0, 0
        for coord in self.test_data:
            target = self.ratings[coord[0]][coord[1]]
            prediction = self.predict(coord[0], coord[1])
            err = abs(target - prediction)
            mae_sum += err
            mse_sum += err * err
        
        test_mae = mae_sum / len(self.test_data)
        test_rmse = sqrt(mse_sum / len(self.test_data))
        self.test_mae.append(test_mae)
        self.test_rmse.append(test_rmse)
        return test_mae, test_rmse
        
        
    def plot_error(self):
        plt.figure()
        plt.suptitle("Factors : {} - LR: {} - VR: {} - BR: {} - PB: {}".format(self.n_factors, self.lr, self.user_vec_reg, self.user_bias_reg, self.pop_bias))
        plt.subplot(211)
        plt.ylabel('RMSE')
        plt.plot(self.test_rmse, 'go', label = "Test")
        plt.plot(self.train_rmse, 'ro', label = "Train")
        plt.legend(loc='lower left')
        plt.subplot(212)
        plt.ylabel('MAE')
        plt.plot(self.test_mae, 'go', label = "Test")
        plt.plot(self.train_mae, 'ro', label = "Train")
        plt.legend(loc='lower left')
        plt.show()
        
    def load_vals(self, dir = "mf_emeddings/"):
        self.user_matrix = pickle.load(open(dir + "user_matrix", "rb" ))
        self.item_matrix = pickle.load(open(dir + "/item_matrix", "rb" ))
        self.user_bias = pickle.load(open(dir + "/user_bias", "rb" ))
        self.item_bias = pickle.load(open(dir + "/item_bias", "rb" ))
            
    def get_user_recs(self, u_id):
        pred_list = np.array([[i_id, self.predict(u_id, i_id)] for i_id in range(self.ratings[u_id].shape[0])])
        pred_list = pred_list[np.isnan(self.ratings[u_id])] # remove previously rated user item interactions
        pred_list = pred_list[pred_list[:,1].argsort()]
        return np.flipud(pred_list)
        
    def get_final_avg_error(self):
        return self.test_rmse[-1]
         
        
# hyperParameterList = { # set hyperparameter combinations
#             "Factors": [5, 10, 50],
#             "Learning Rate": [.01, .001],
#             "Populate Bias": [True, False],
#             "Bias Regs": [0, .01, .1],
#             "Vector Regs": [0, .01, .1],
#             "Epochs": [1, 3],
#             }

def test_params():
    ratings = data.get_data("ml-latest-small/ratings.csv")
    hyperParameterList = { # set hyperparameter combinations
                "Factors": [10, 50, 200],
                "Learning Rate": [.01, .001],
                "Populate Bias": [True],
                "Bias Regs": [.1, .2, .5],
                "Vector Regs": [.1, .2, .5],
                "Epochs": [3],
                }
    
    keys, values = zip(*hyperParameterList.items())
    min_error = np.inf
    
    for v in product(*values):
        e = dict(zip(keys, v)) # generate hyperparameter combination
        print("Using hyperparams: {0}".format(e))
        my_sgdMF = sgdMF(ratings.to_numpy(), e["Factors"], lr = e["Learning Rate"], populate_bias = e["Populate Bias"], user_vec_reg = e["Vector Regs"], item_vec_reg =  e["Vector Regs"], user_bias_reg = e["Bias Regs"], item_bias_reg = e["Bias Regs"])
        
        for i in range(e["Epochs"]):
            my_sgdMF.train()
        my_sgdMF.plot_error()
        
        if my_sgdMF.min_error < min_error:
            min_error = my_sgdMF.min_error
            best_params = e
            best_user_matrix, best_item_matrix, best_user_bias, best_item_bias = my_sgdMF.best_user_matrix, my_sgdMF.best_item_matrix, my_sgdMF.best_user_bias, my_sgdMF.best_item_bias
            print("New best params: {0}".format(best_params))
    
    print("Best parameters: {0}, RMSE: {1}".format(best_params, round(min_error, 3)))
    pickle.dump(best_user_matrix, open("mf_emeddings/user_matrix", "wb"))
    pickle.dump(best_item_matrix, open("mf_emeddings/item_matrix", "wb" ))
    pickle.dump(best_user_bias, open("mf_emeddings/user_bias", "wb" ))
    pickle.dump(best_item_bias, open("mf_emeddings/item_bias", "wb" ))
    


def load_predictions(u_id = 1, n = 10, verbose = True):
    ratings = data.get_data("ml-latest-small/ratings.csv")
    my_sgdMF = sgdMF(ratings.to_numpy())
    recs = my_sgdMF.get_user_recs(u_id)
    
    if n:
        recs = recs[:n]
    
    if verbose:
        movies = data.get_data_raw("ml-latest-small/movies.csv").to_numpy()
        print("Reccomendations for user {}:\n".format(u_id))
        for i, m in enumerate(recs):
            print("{}. {} - {}".format(i + 1, movies[int(m[0])][1], str(round(m[1], 4))))
            
    return recs

#test_params()
#load_predictions(10, 0)
    
    
# my_sgdMF = sgdMF(ratings.to_numpy(), 20, lr = .001, populate_bias = False, user_bias_reg = 0, item_bias_reg = 0)
# for i in range(3):
#     my_sgdMF.train()
# my_sgdMF.plot_error()


#https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea
#https://medium.com/coinmonks/how-to-implement-a-recommendation-system-with-deep-learning-and-pytorch-2d40476590f9

    

    