# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:39:48 2021

@author: sgawalsh
"""
import pandas

def get_data(path, sep = '::'):
    try:
        ratings = pandas.read_csv(path)
    except:
        ratings = pandas.read_table(path, sep=sep, header = None)
        ratings = ratings.rename(columns={0: "userId", 1: "movieId", 2: "rating"})
    
    return ratings.pivot(index = 'userId', columns = 'movieId', values = 'rating')

def get_data_raw(path, sep = '::'):
    try:
        return pandas.read_csv(path)
    except:
        return pandas.read_table(path, sep=sep, header = None)