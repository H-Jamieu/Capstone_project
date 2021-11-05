import os
import pandas as pd
import numpy as np


def build_rating_matrix(df):
    """
    for building matrix in movie - rating - user form
    :param df:
    :return:
    """
    dummy_dict = {}
    dummy_list = []
    movie_list = df['movieId'].unique()
    user_list = df['userId'].unique()
    for i in range(0,len(user_list)):
        dummy_list.append(0)
    for m in movie_list:
        dummy_dict[str(m)] = dummy_list
    df_mu = pd.DataFrame(data=dummy_dict,index=movie_list)
    for idx in df.index:
        df_mu.at[df.loc[idx, 'userId'], str(df.loc[idx, 'movieId'])] = df.loc[idx, 'rating']
    return df_mu





