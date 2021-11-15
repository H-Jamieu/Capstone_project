import pandas as pd
import numpy as np

from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from tqdm import tqdm
import gc
def reduce_mem(df, cols):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in tqdm(cols):
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


flag = 'syn'
TRAIN_PATH = 'Data/User_data/train_'+flag+'_5000.csv'
TEST_PATH = 'Data/User_data/test_'+flag+'_5000.csv'
ALL_unwatched = 'Data/Predictions/out_'+flag+'_5000.csv'
EMBEDDING = {1:'Data/Movie_data/full_embedding.csv',2:'Data/Movie_data/genome74.csv'}
choice = 2

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
embedding = pd.read_csv(EMBEDDING[choice])
embedding.set_index('movieId')

ratingcols = ['rating']
timecols = ['timestamp']
idcols = ['userId','movieId']
#df.drop(df.filter(regex="Unname"),axis=1, inplace=True)
def MSE(actual, pred):
    return(((actual-pred)**2).sum()/len(actual))

long_train = train.merge(embedding, on='movieId', how='left')
long_test = test.merge(embedding, on='movieId', how='left')
cols = [f for f in long_test.columns if f not in ['date_','Unnamed: 0'] + timecols + ratingcols]
long_train = reduce_mem(long_train,long_train.columns)
long_test = reduce_mem(long_test,long_test.columns)
embedding = reduce_mem(embedding,embedding.columns)


LGBM = LGBMRegressor(
        learning_rate=0.05,
        n_estimators=800,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=2021
    )
LGBM.fit(
        long_train[cols], long_train[ratingcols],
        eval_set=[(long_train[cols], long_train[ratingcols])],
        early_stopping_rounds=8,
        verbose=100,
    )
long_test['pred_rating'] = LGBM.predict(long_test[cols])
print(MSE(long_test['rating'],long_test['pred_rating']))
out_test = long_test[['movieId','userId','pred_rating']]
out_test.to_csv('Data/Predictions/out_LGBM_test_'+flag+str(choice)+'_5000.csv')
XGB = XGBRegressor(max_depth=16,
                    n_estimators =70)

XGB.fit(
        long_train[cols], long_train[ratingcols],
        eval_set=[(long_train[cols], long_train[ratingcols])],
        early_stopping_rounds=3,
        verbose=10)

long_test['pred_rating_XGB'] = XGB.predict(long_test[cols])
print(MSE(long_test['rating'],long_test['pred_rating_XGB']))
out_test = long_test[['movieId','userId','pred_rating_XGB']]
out_test.to_csv('Data/Predictions/out_XGB_test_'+flag+str(choice)+'_5000.csv')
del(long_train)
del(long_test)

out_cols = ['userId','movieId','rating']
out_table = pd.read_csv(ALL_unwatched)[idcols]
out_table = reduce_mem(out_table,out_table.columns)
long_out_table = out_table.merge(embedding, on='movieId', how='left')
del(out_table)
long_out_table = reduce_mem(long_out_table,cols)
long_out_table['rating'] = LGBM.predict(long_out_table[cols])
out_table = long_out_table[out_cols]
out_table.to_csv('Data/Predictions/out_LGBM_'+flag+str(choice)+'_5000.csv')
out_colsx = ['userId','movieId','rating_XGB']
long_out_table['rating_XGB'] = XGB.predict(long_out_table[cols])
del(out_table)
out_table = long_out_table[out_colsx]
out_table.to_csv('Data/Predictions/out_XGB_'+flag+str(choice)+'_5000.csv')