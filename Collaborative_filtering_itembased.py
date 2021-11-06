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
  return tf.SparseTensor(
      indices=indices,
      values=values,
      dense_shape=[users.shape[0], movies.shape[0]])

def sparse_mean_square_error(sparse_ratings, user_embeddings, movie_embeddings):
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
      "userId": U,
      "movieId": V
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
  scores = V.dot(u.T)
  return scores

def user_recommendations(model, mid,measure=DOT):
    scores = compute_scores(
        model.embeddings["userId"], model.embeddings["movieId"][mid])
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

def normalize(rl):
    minum = min(rl)
    maxim = max(rl)
    normed = [5*(r-minum)/(maxim-minum) for r in rl]
    return normed


def generate_recommendation(model, users, full_rate, movies):
    # Users and movies kept the original order in the dataframe
    # user, movie ,rating format
    rating_opt = []
    for m in range(0,len(movies)):
        ratings = normalize(user_recommendations(model, m))
        rated = full_rate[full_rate.movieId == m]['userId'].values
        for r in range(0, len(ratings)):
            if users[r] not in rated:
                rating_opt.append([m, users[r], ratings[r]])
    return rating_opt

def to_output(list_in):
    first_row = ['movieId','userId','rating']
    with open('Data/Predictions/out_syn_5000.csv', 'w', newline='') as pred:
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

train_ratings, test_ratings = pd.read_csv('Data/User_data/train_syn_5000.csv'), pd.read_csv('Data/User_data/test_syn_5000.csv')
users = train_ratings['userId'].unique()
movies = train_ratings['movieId'].unique()


model = build_model(train_ratings, test_ratings, embedding_dim=100, init_stddev=0.6)
model.train(num_iterations=20000, learning_rate=8.)
output = generate_recommendation(model, users, train_ratings, movies)
print(output[:20])
to_output(output)





