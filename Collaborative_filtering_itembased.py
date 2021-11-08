from __future__ import print_function

import numpy as np
import pandas as pd
import collections
from mpl_toolkits.mplot3d import Axes3D
from IPython import display
from matplotlib import pyplot as plt
import sklearn
import sklearn.manifold
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)
import os
import csv
from tqdm import tqdm


DICT_M = {}
DICT_U = {}
DICT_MR = {}
DICT_UR = {}

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

def map_index(movies, users):
    seq_m = list(np.arange(0, len(movies)))
    seq_u = list(np.arange(0, len(users)))
    dict_m = dict(zip(movies, seq_m))
    dict_u = dict(zip(users, seq_u))
    dict_mr = dict(zip(seq_m, movies))
    dict_ur = dict(zip(seq_u, users))
    return dict_m, dict_u, dict_mr, dict_ur

def build_rating_sparse_tensor(ratings_df):
  """
  Args:
    ratings_df: a pd.DataFrame with `user_id`, `movie_id` and `rating` columns.
  Returns:
    a tf.SparseTensor representing the ratings matrix.
  """
  indices = ratings_df[['movieId', 'userId']].values
  values = ratings_df['rating'].values
  movies = ratings_df['movieId'].unique()
  users = ratings_df['userId'].unique()
  for i in indices:
      i[0] = DICT_M[i[0]]
      i[1] = DICT_U[i[1]]
  return tf.SparseTensor(
      indices=indices,
      values=values,
      dense_shape=[movies.shape[0],users.shape[0]])

def sparse_mean_square_error(sparse_ratings, user_embeddings,movie_embeddings):
  """
  Args:
    sparse_ratings: A SparseTensor rating matrix, of dense_shape [N, M]
    user_embeddings: A dense Tensor U of shape [N, k] where k is the embedding
      dimension, such that U_i is the embedding of user i.
    movie_embeddings: A dense Tensor V of shape [M, k] where k is the embedding
      dimension, such that V_j is the embedding of movie j.
  Returns:
    A scalar Tensor representing the MSE between the true ratings and the
      model's predictions.
  """
  predictions = tf.gather_nd(
      tf.matmul(movie_embeddings, user_embeddings, transpose_b=True),
      sparse_ratings.indices)
  # predictions = tf.reduce_sum(
  #     tf.gather(movie_embeddings, sparse_ratings.indices[:, 0]) *
  #     tf.gather(user_embeddings, sparse_ratings.indices[:, 1]),
  #     axis=1)
  loss = tf.losses.mean_squared_error(sparse_ratings.values, predictions)
  return loss

class CFModel(object):
  """Simple class that represents a collaborative filtering model"""
  def __init__(self, embedding_vars, loss, metrics=None):
    """Initializes a CFModel.
    Args:
      embedding_vars: A dictionary of tf.Variables.
      loss: A float Tensor. The loss to optimize.
      metrics: optional list of dictionaries of Tensors. The metrics in each
        dictionary will be plotted in a separate figure during training.
    """
    self._embedding_vars = embedding_vars
    self._loss = loss
    self._metrics = metrics
    self._embeddings = {k: None for k in embedding_vars}
    self._session = None

  @property
  def embeddings(self):
    """The embeddings dictionary."""
    return self._embeddings

  def train(self, num_iterations=100, learning_rate=1.0, plot_results=True,
            optimizer=tf.train.GradientDescentOptimizer):
    """Trains the model.
    Args:
      iterations: number of iterations to run.
      learning_rate: optimizer learning rate.
      plot_results: whether to plot the results at the end of training.
      optimizer: the optimizer to use. Default to GradientDescentOptimizer.
    Returns:
      The metrics dictionary evaluated at the last iteration.
    """
    with self._loss.graph.as_default():
      opt = optimizer(learning_rate)
      train_op = opt.minimize(self._loss)
      local_init_op = tf.group(
          tf.variables_initializer(opt.variables()),
          tf.local_variables_initializer())
      if self._session is None:
        self._session = tf.Session()
        with self._session.as_default():
          self._session.run(tf.global_variables_initializer())
          self._session.run(tf.tables_initializer())
          tf.train.start_queue_runners()

    with self._session.as_default():
      local_init_op.run()
      iterations = []
      metrics = self._metrics or ({},)
      metrics_vals = [collections.defaultdict(list) for _ in self._metrics]

      # Train and append results.
      for i in range(num_iterations + 1):
        _, results = self._session.run((train_op, metrics))
        if (i % 10 == 0) or i == num_iterations:
          print("\r iteration %d: " % i + ", ".join(
                ["%s=%f" % (k, v) for r in results for k, v in r.items()]),
                end='')
          iterations.append(i)
          for metric_val, result in zip(metrics_vals, results):
            for k, v in result.items():
              metric_val[k].append(v)

      for k, v in self._embedding_vars.items():
        self._embeddings[k] = v.eval()

      if plot_results:
        # Plot the metrics.
        num_subplots = len(metrics)+1
        fig = plt.figure()
        fig.set_size_inches(num_subplots*10, 8)
        for i, metric_vals in enumerate(metrics_vals):
          ax = fig.add_subplot(1, num_subplots, i+1)
          for k, v in metric_vals.items():
            ax.plot(iterations, v, label=k)
          ax.set_xlim([1, num_iterations])
          ax.legend()
      return results

def build_model(train, test, embedding_dim=3, init_stddev=1.):
  """
  Args:
    ratings: a DataFrame of the ratings
    embedding_dim: the dimension of the embedding vectors.
    init_stddev: float, the standard deviation of the random initial embeddings.
  Returns:
    model: a CFModel.
  """
  # Split the ratings DataFrame into train and test.
  train_ratings, test_ratings = train, test
  # SparseTensor representation of the train and test datasets.
  A_train = build_rating_sparse_tensor(train_ratings)
  A_test = build_rating_sparse_tensor(test_ratings)
  # Initialize the embeddings using a normal distribution.
  U = tf.Variable(tf.random_normal(
      [A_train.dense_shape[0], embedding_dim], stddev=init_stddev))
  V = tf.Variable(tf.random_normal(
      [A_train.dense_shape[1], embedding_dim], stddev=init_stddev))
  train_loss = sparse_mean_square_error(A_train, U, V)
  test_loss = sparse_mean_square_error(A_test, U, V)
  metrics = {
      'train_error': train_loss,
      'test_error': test_loss
  }
  embeddings = {
      "movieId": U,
      "userId": V
  }
  return CFModel(embeddings, train_loss, [metrics])

DOT = 'dot'
COSINE = 'cosine'
def compute_scores(query_embedding, item_embeddings, measure=DOT):
  """Computes the scores of the candidates given a query.
  Args:
    query_embedding: a vector of shape [k], representing the query embedding.
    item_embeddings: a matrix of shape [N, k], such that row i is the embedding
      of item i.
    measure: a string specifying the similarity measure to be used. Can be
      either DOT or COSINE.
  Returns:
    scores: a vector of shape [N], such that scores[i] is the score of item i.
  """
  u = query_embedding
  V = item_embeddings
  if measure == COSINE:
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    u = u / np.linalg.norm(u)
  scores = u.dot(V.T)
  return scores

def user_recommendations(model,measure=COSINE):
    scores = compute_scores(
        model.embeddings["movieId"], model.embeddings["userId"])
    return scores
    # if exclude_rated:
    #     # remove movies that are already rated
    #     rated_movies = ratings[ratings.user_id == "943"]["movie_id"].values
    #     df = df[df.movie_id.apply(lambda movie_id: movie_id not in rated_movies)]
    # display.display(df.sort_values([score_key], ascending=False).head(k))

#development target:
# 1. output a predicted rating of user for their unwatched movies (scaled into (0,5) range)
# 2. output a mean square error for test set so that we can have an inference of rating
# 3. change the input into scaled form

def denormalize(rl, mid, ratings):
    #Relu-like normlization
    normed = []
    maxr = ratings.query('movieId=='+str(mid))['movie_rating_max'].iloc[0]
    minr = ratings.query('movieId==' + str(mid))['movie_rating_min'].iloc[0]
    avgr = ratings.query('movieId==' + str(mid))['movie_rating_mean'].iloc[0]
    for r in rl:
        rnormed = 5*r+avgr
        if rnormed>maxr:
            rnormed=maxr
        if rnormed<minr:
            rnormed = minr
        normed.append(rnormed)
    return normed


def naive_normlize(scores):
    maxs = max(scores)
    mins = min(scores)
    normed = [ 5*(s-mins)/(maxs-mins) for s in scores]
    return normed


def generate_recommendation(model,  full_rate):
    # Users and movies kept the original order in the dataframe
    # user, movie ,rating format
    rating_opt = []
    ratings = user_recommendations(model)
    for row in tqdm(range(0, len(ratings))):
        mid = DICT_MR[row]
        rated = full_rate[full_rate.movieId == mid]['userId'].values
        normed_scores = naive_normlize(ratings[row])
        for u in range(0 ,len(ratings[row])):
            uid = DICT_UR[u]
            if u not in rated:
                rating_opt.append([mid, uid, normed_scores[u]])
        # for r in range(0, len(ratings)):
        #     uid = DICT_UR[r]
        #     if uid not in rated:
        #         rating_opt.append([mid, uid, ratings[r]])
    return rating_opt

def gravity(U, V):
  """Creates a gravity loss given two embedding matrices."""
  return 1. / (U.shape[0].value*V.shape[0].value) * tf.reduce_sum(
      tf.matmul(U, U, transpose_a=True) * tf.matmul(V, V, transpose_a=True))

def build_regularized_model(
    train ,test, embedding_dim=3, regularization_coeff=.1, gravity_coeff=1.,
    init_stddev=0.1):
  """
  Args:
    ratings: the DataFrame of movie ratings.
    embedding_dim: The dimension of the embedding space.
    regularization_coeff: The regularization coefficient lambda.
    gravity_coeff: The gravity regularization coefficient lambda_g.
  Returns:
    A CFModel object that uses a regularized loss.
  """
  # Split the ratings DataFrame into train and test.
  train_ratings, test_ratings = train, test
  # SparseTensor representation of the train and test datasets.
  A_train = build_rating_sparse_tensor(train_ratings)
  A_test = build_rating_sparse_tensor(test_ratings)
  U = tf.Variable(tf.random_normal(
      [A_train.dense_shape[0], embedding_dim], stddev=init_stddev))
  V = tf.Variable(tf.random_normal(
      [A_train.dense_shape[1], embedding_dim], stddev=init_stddev))

  error_train = sparse_mean_square_error(A_train, U, V)
  error_test = sparse_mean_square_error(A_test, U, V)
  gravity_loss = gravity_coeff * gravity(U, V)
  regularization_loss = regularization_coeff * (
      tf.reduce_sum(U*U)/U.shape[0].value + tf.reduce_sum(V*V)/V.shape[0].value)
  total_loss = error_train + regularization_loss + gravity_loss
  losses = {
      'train_error_observed': error_train,
      'test_error_observed': error_test,
  }
  loss_components = {
      'observed_loss': error_train,
      'regularization_loss': regularization_loss,
      'gravity_loss': gravity_loss,
  }
  embeddings = {"movieId": U, "userId": V}

  return CFModel(embeddings, total_loss, [losses, loss_components])

def to_output(list_in,flag,mod):
    first_row = ['movieId','userId','rating']
    with open('Data/Predictions/out_'+flag+'_'+mod+'_'+'_5000.csv', 'w', newline='') as pred:
        wr = csv.writer(pred)
        wr.writerow(first_row)
        wr.writerows(list_in)
    pred.close()

def stat_anla(scores):
    maxim = max(scores)
    minum = min(scores)
    average = sum(scores)/len(scores)
    print('Max is '+ str(maxim))
    print('Min is '+ str(minum))
    print('Avergae is '+ str(average))




flag = 'top'
mod = 'reg'
train_ratings, test_ratings = pd.read_csv('Data/User_data/train_'+flag+'_5000.csv'), pd.read_csv('Data/User_data/test_'+flag+'_5000.csv')
train_ratings = train_ratings.sort_values(by =['movieId','userId'])
test_ratings = test_ratings.sort_values(by=['movieId','userId'])
train_ratings['movie_rating_mean'] = train_ratings.groupby('movieId')['rating'].transform('mean')
train_ratings['movie_rating_max'] = train_ratings.groupby('movieId')['rating'].transform('max')
train_ratings['movie_rating_min'] = train_ratings.groupby('movieId')['rating'].transform('min')
train_ratings['normalized_rating'] = (train_ratings['rating'] - train_ratings['movie_rating_mean'])/5
test_ratings['movie_rating_mean'] = test_ratings.groupby('movieId')['rating'].transform('mean')
test_ratings['movie_rating_max'] = test_ratings.groupby('movieId')['rating'].transform('max')
test_ratings['movie_rating_min'] = test_ratings.groupby('movieId')['rating'].transform('min')
test_ratings['normalized_rating'] = (test_ratings['rating'] - test_ratings['movie_rating_mean'])/5
users = train_ratings['userId'].unique()
movies = train_ratings['movieId'].unique()

DICT_M, DICT_U, DICT_MR, DICT_UR = map_index(movies, users)

# sparse_test = build_rating_sparse_tensor(test_ratings)
# with tf.Session() as sess:
#     sess.run(sparse_test) #execute init_op
#     #print the random values that we sample
#     print (sess.run(sparse_test))

model = build_regularized_model(train_ratings, test_ratings, embedding_dim=36, init_stddev=0.6)
model.train(num_iterations=24000, learning_rate=0.4)
output = generate_recommendation(model,  train_ratings)
print(output[:20])
to_output(output, flag, mod)





