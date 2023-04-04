import pprint

import numpy as np
import pandas as pd 

import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs



class RankingModel(tfrs.Model):

  def __init__(self, loss, unique_query, unique_data):
    super().__init__()
    embedding_dimension = 32

    # Compute embeddings for users.
    self.query_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_query),
      tf.keras.layers.Embedding(len(unique_query) + 2, embedding_dimension)
    ])

    # Compute embeddings for movies.
    self.data_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_data),
      tf.keras.layers.Embedding(len(unique_data) + 2, embedding_dimension)
    ])

    # Compute predictions.
    self.score_model = tf.keras.Sequential([
      # Learn multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      # Make rating predictions in the final layer.
      tf.keras.layers.Dense(1)
    ])

    self.task = tfrs.tasks.Ranking(
      loss=loss,
      metrics=[
        tfr.keras.metrics.NDCGMetric(name="ndcg_metric"),
        tf.keras.metrics.RootMeanSquaredError()
      ]
    )

  def call(self, features):
    # We first convert the id features into embeddings.
    # User embeddings are a [batch_size, embedding_dim] tensor.
    query_embeddings = self.query_embeddings(features["query"])

    # Movie embeddings are a [batch_size, num_movies_in_list, embedding_dim]
    # tensor.
    data_embeddings = self.data_embeddings(features["data"])

    # We want to concatenate user embeddings with movie emebeddings to pass
    # them into the ranking model. To do so, we need to reshape the user
    # embeddings to match the shape of movie embeddings.
#     print("list_length=========================>", list_length)
    list_length = features["data"].shape[1]
    data_embedding_repeated = tf.repeat(
        tf.expand_dims(query_embeddings, 1), [list_length], axis=1)

    # Once reshaped, we concatenate and pass into the dense layers to generate
    # predictions.
    concatenated_embeddings = tf.concat(
        [data_embedding_repeated, data_embeddings], 2)

    return self.score_model(concatenated_embeddings)

  def compute_loss(self, features, training=False):
    labels = features.pop("revelvence")

    scores = self(features)

    return self.task(
        labels=labels,
        predictions=tf.squeeze(scores, axis=-1),
    )



        