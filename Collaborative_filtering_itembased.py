import os
import pandas as pd
import torch
import numpy as np


def build_rating_matrix(df, scale = 0):
    """
    for building matrix in movie - rating - user form
    :param df:
    :return:
    """
    dummy_dict = {}
    movie_list = df['movieId'].unique()
    user_list = df['userId'].unique()
    dummy_list = [float(0)]*len(movie_list)

    scale_score(df, scale)

    for u in user_list:
        dummy_dict[str(u)] = dummy_list
    df_mu = pd.DataFrame(data=dummy_dict, index=movie_list)

    for idx in df.index:
        df_mu.at[df.loc[idx, 'movieId'], str(df.loc[idx, 'userId'])] = float((df.loc[idx, 'rating'] - df.loc[idx, 'avg_rating'])/5)
    return df_mu

def scale_score(df, flag= 0):
    if flag == 0:
        df['avg_rating'] = 0
    else:
        df['avg_rating'] = df.groupby('movieId')['rating'].transform('mean')

def cosine_simularity():
    return 0

df1 = pd.read_csv('Data/User_data/test_top_5000.csv')
print(build_rating_matrix(df1,1).head())





