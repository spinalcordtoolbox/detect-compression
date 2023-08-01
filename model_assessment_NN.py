import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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


path_to_csv = '/home/GRAMES.POLYMTL.CA/p118968/data_nvme_p118968/train_test/fichier_fusionne.csv'

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

features_selection_manual = ['area', 'diameter_AP', 'solidity', 'CompressionRatio', 'diameter_RL', 'orientation', 'Torsion', 'VertLevel']
selected_features = features_selection_manual

# adjust dataset with respect to feature selection
X_train = X_train.drop(columns=[col for col in X_train.columns if col not in selected_features])
X_test = X_test.drop(columns=[col for col in X_test.columns if col not in selected_features])

X_train_csv = '/home/GRAMES.POLYMTL.CA/p118968/data_nvme_p118968/train_test/X_train.csv'
X_test_csv  = '/home/GRAMES.POLYMTL.CA/p118968/data_nvme_p118968/train_test/X_test.csv'

# Écrire le DataFrame dans le fichier CSV
X_train.to_csv(X_train_csv, index=False)
X_test.to_csv(X_test_csv, index=False)



input_dim = np.shape(X_train)[1]

# Define the model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    layers.Dropout(0.2),  # Dropout layer with a dropout rate of 0.2
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),  # Dropout layer with a dropout rate of 0.2
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics='roc_auc_score')

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=16)

tf.saved_model.save(model, "/home/GRAMES.POLYMTL.CA/p118968/data_nvme_p118968/resultats/")



