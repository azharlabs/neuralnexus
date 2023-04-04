import pandas as pd
from sklearn.datasets import load_breast_cancer

import sys
sys.path.insert(0, 'E://neuralnexus')
print(sys.path)

import tensorflow as tf

import neuralnexus as nn

from sklearn.model_selection import train_test_split
from neuralnexus.ranking import RankingModel

from neuralnexus.ranking.utils import get_unique_data, get_unique_query, dataPrep
from neuralnexus.ranking.losses import ListMLELoss

df = pd.read_csv('topic_ranking_data_prep.csv')




unique_data = get_unique_data(df, 'data')

unique_query = get_unique_query(df, 'query')

data = dataPrep(df)

epochs = 10 

cached_train = data.shuffle(100_000).batch(8192).cache()


#list wise sorting 

listwise_model = RankingModel(ListMLELoss(), unique_query, unique_data)
listwise_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

# training
listwise_model.fit(cached_train, epochs=epochs, verbose=False)

# tesing
listwise_model_result = listwise_model.evaluate(cached_train, return_dict=True)
print("NDCG of the ListMLE model: {:.4f}".format(listwise_model_result["ndcg_metric"]))


