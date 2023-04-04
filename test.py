# Importing the Packages:
import optuna
import pandas as pd
from sklearn import linear_model
from sklearn import ensemble
from sklearn import datasets
from sklearn import model_selection

#Grabbing a sklearn Classification dataset:
X,y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)

#Step 1. Define an objective function to be maximized.
def objective(trial):

    classifier_name = trial.suggest_categorical("classifier", ["LogReg", "RandomForest"])
    
    # Step 2. Setup values for the hyperparameters:
    if classifier_name == 'LogReg':
        logreg_c = trial.suggest_float("logreg_c", 1e-10, 1e10, log=True)
        classifier_obj = linear_model.LogisticRegression(C=logreg_c)
    else:
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 1000)
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = ensemble.RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=rf_n_estimators
        )

    # Step 3: Scoring method:
    score = model_selection.cross_val_score(classifier_obj, X, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy

# Step 4: Running it
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)



array([[array([b'Postman, The (1997)', b'Liar Liar (1997)', b'Contact (1997)',
               b'Welcome To Sarajevo (1997)',
               b'I Know What You Did Last Summer (1997)'], dtype=object)      ,
        b'681', array([4., 5., 1., 4., 1.], dtype=float32)],])

array([{'user_id': b'681', 'movie_title': array([b'Postman, The (1997)', b'Liar Liar (1997)', b'Contact (1997)',
              b'Welcome To Sarajevo (1997)',
              b'I Know What You Did Last Summer (1997)'], dtype=object), 'user_rating': array([4., 5., 1., 4., 1.], dtype=float32)},
])