import numpy as np 
import pandas as pd 

import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs



def get_unique_data(df, column):
  """
  `get_unique_data`  take two arguments, dataframe and column. 

  ex:- 

  unique_data = get_unique_data(df, column='data')

  input:-
  [[topic, topic1, topic2], [topic2, topic3, topic1]] 

  Output:-
  [topic, topic1, topic2, topic3]

  get_unique_data will return the unique values of the list of list 
  """
  all_data = []

  for _lst in df[column]:
      all_data = all_data + eval(_lst)
   
  return np.unique(all_data)
    

def get_unique_query(df, column):
  """
  `get_unique_query`  take two arguments, dataframe and column. 

  ex:- 

  unique_query = get_unique_query(df, column='query')

  input:-
  [topic, topic1, topic2, topic2, topic3, topic1] 

  Output:-
  [topic, topic1, topic2, topic3]

  get_unique_query will return the unique values of the list
  """
  return np.unique(list(df[column]))


def dataPrep(df):
  """ 
  dataPrep is convert the dataframe to tf.tensor

  ex:-
  tensor = dataPrep(df)

  condition:- 
  dataframe should have columns in the below given order.

  columns = ['data', 'query', 'revelence']

  data = [topic, topic1, topic2, topic3]

  query = 'name'

  revelence = [0.4, 0.5, 0.9, 0.2]

  """ 
  df.columns = ['data', 'query', 'revelence']

  list_of_data = [eval(i) for i in df['data']]
  list_of_revelence = [eval(i) for i in df['revelence']]

  data = list_of_data
  query = list(df['query'])
  revelence = list_of_revelence

  # convert NumPy arrays to TensorFlow tensors
  data_tensor = tf.convert_to_tensor(data)
  query_tensor = tf.convert_to_tensor(query)
  revelence_tensor = tf.convert_to_tensor(revelence)
  revelence_tensor = tf.cast(revelence_tensor, tf.float32)

  # create a new dictionary with the converted tensors
  my_new_dict = {'data': data_tensor, 
              'query': query_tensor, 
              'revelvence': revelence_tensor}

  # create the TensorSliceDataset
  data = tf.data.Dataset.from_tensor_slices(my_new_dict)
  
  return data