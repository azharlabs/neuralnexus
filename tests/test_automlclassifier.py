import pandas as pd
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from neuralnexus.automl.classifier import Model




def test_classifer():
    d = load_breast_cancer()
    y = d['target']
    X = pd.DataFrame(d['data'],columns = d['feature_names'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    model = Model()
    model.fit(X_train,y_train)
    assert model.model_status == 200