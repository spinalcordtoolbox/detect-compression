import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from copy import deepcopy
import time
#import shap


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import make_scorer, roc_auc_score



import seaborn as sns
import xgboost as xgb


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

path_to_csv = 'fichier_fusionne.csv'

df = pd.read_csv(path_to_csv)

#print(df.info)

# data cleaning 
df = df.dropna(axis=0)

# divide the dataset 
y = df['is_compressed']
X = df.drop(columns='is_compressed')

# reverse sigmoid: transform the non linear regression into a linear regression

# Add a constant column for the intercept term
X['intercept'] = 1

# Create an instance of the StandardScaler
#scaler = StandardScaler()
scaler = MinMaxScaler()



# Apply the scaling transformation to both the training and test data
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)



# Create the SVM classifier
clf = SVC()

# Define the parameter grid for grid search

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto'] + [10**i for i in range(-5, 4)],
    'coef0': [0, 0.5, 1, 2],
    'shrinking': [True, False],
    'probability': [True, False],
    'tol': [1e-5, 1e-4, 1e-3, 1e-2],
    'decision_function_shape': ['ovo', 'ovr'],
    'break_ties': [True, False],
    'cache_size': [100, 500, 1000],
    'max_iter': [100, 500, 1000],
    'random_state': [42],
    'class_weight': [None, 'balanced', {0: 1, 1: 2}],
    'verbose': [True, False],
}

# Perform grid search with verbose output
grid_search = GridSearchCV(clf, param_grid, scoring='accuracy', verbose=10)
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params_svc = grid_search.best_params_
best_score_svc = grid_search.best_score_

# Formater les résultats sous forme de texte
resultats_texte = f"Meilleurs paramètres : {best_params_svc}\nScore du meilleur modèle : {best_score_svc}"

# Nom du fichier dans lequel vous voulez enregistrer les résultats
nom_fichier = "resultats/SVM.txt"

# Enregistrer les résultats dans un fichier texte
with open(nom_fichier, "w") as fichier:
    fichier.write(resultats_texte)

