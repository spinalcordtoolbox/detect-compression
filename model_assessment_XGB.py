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

'''
X_train = pd.read_csv('/home/GRAMES.POLYMTL.CA/p118968/data_nvme_p118968/train_test/X_train.csv')
X_test  = pd.read_csv('/home/GRAMES.POLYMTL.CA/p118968/data_nvme_p118968/train_test/X_test.csv')
y_train = pd.read_csv('/home/GRAMES.POLYMTL.CA/p118968/data_nvme_p118968/train_test/y_train.csv')
y_test = pd.read_csv('/home/GRAMES.POLYMTL.CA/p118968/data_nvme_p118968/train_test/y_test.csv')
'''

path_to_csv = '/home/GRAMES.POLYMTL.CA/p118968/data_nvme_p118968/train_test/fichier_fusionne_2.csv'

df = pd.read_csv(path_to_csv)

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

features_selection_manual = ['area', 'diameter_AP', 'solidity', 'CompressionRatio', 'diameter_RL', 'orientation', 'Torsion', 'VertLevel']
selected_features = features_selection_manual

# adjust dataset with respect to feature selection
X_train = X_train.drop(columns=[col for col in X_train.columns if col not in selected_features])
X_test = X_test.drop(columns=[col for col in X_test.columns if col not in selected_features])

X_train_csv = '/home/GRAMES.POLYMTL.CA/p118968/data_nvme_p118968/train_test/X_train_2.csv'
X_test_csv  = '/home/GRAMES.POLYMTL.CA/p118968/data_nvme_p118968/train_test/X_test_2.csv'
y_train_csv = '/home/GRAMES.POLYMTL.CA/p118968/data_nvme_p118968/train_test/y_train_2.csv'
y_test_csv = '/home/GRAMES.POLYMTL.CA/p118968/data_nvme_p118968/train_test/y_test_2.csv'


# Écrire le DataFrame dans le fichier CSV
X_train.to_csv(X_train_csv, index=False)
X_test.to_csv(X_test_csv, index=False)
y_train.to_csv(y_train_csv, index=False)
y_test.to_csv(y_test_csv, index=False)


param_grid = {
    'learning_rate': [0.25, 0.3, 0.35],
    'max_depth': [40, 35, 45],
    'n_estimators': [550, 525, 575],
    'gamma': [0, 0.01],
    'subsample': [0.85, 0.9, 0.95],
    'colsample_bytree': [0.9, 1.0],
    'reg_alpha': [0.15, 0.2, 0.25],
    'reg_lambda': [0.3, 0.4, 0.5]
}

# Définition du modèle XGBoost
model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

scoring = make_scorer(roc_auc_score, greater_is_better=True)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=10, verbose=10, n_jobs=-1)

# Exécution du GridSearchCV
grid_search.fit(X_train, y_train)

# Récupération des meilleurs hyperparamètres
best_params_xgb = grid_search.best_params_
best_score_xgb = grid_search.best_score_

print(best_params_xgb, best_score_xgb)

# Formater les résultats sous forme de texte
resultats_texte = f"Meilleurs paramètres : {best_params_xgb}\nScore du meilleur modèle : {best_score_xgb}"

# Nom du fichier dans lequel vous voulez enregistrer les résultats
nom_fichier = "/home/GRAMES.POLYMTL.CA/p118968/data_nvme_p118968/resultats/xgb_w_s_generic.txt"

# Enregistrer les résultats dans un fichier texte
with open(nom_fichier, "w") as fichier:
    fichier.write(resultats_texte)


