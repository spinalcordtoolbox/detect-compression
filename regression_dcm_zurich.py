import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

path_to_csv = '/Users/etiennedufayet/Desktop/STAGE_3A/Compression_detection_zurich/dataset_zurich_metrics.csv'

df = pd.read_csv(path_to_csv)

print(df.info)

# data cleaning 
df = df.drop(columns=['DistancePMJ', 'VertLevel'])
df = df.dropna(axis=0)

# divide the dataset 
y = df['is_compressed']
X = df.drop(columns='is_compressed')

# reverse sigmoid: transform the non linear regression into a linear regression

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Add a constant column for the intercept term
X['intercept'] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)


model = LogisticRegression()

# Perform forward stepwise selection
selector = SequentialFeatureSelector(model, direction='forward', n_features_to_select='auto', tol=None)
selector.fit(X_train, y_train)

# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)
params = selector.get_params(deep=True)


selected_features = X_train.columns[selected_feature_indices]


# Print the selected features and params
'''
print("Selected Features:", selected_features)
print(params)
'''


# adjust columns with respect to feature selection
X_train = X_train.drop(columns = selected_features)
X_test = X_test.drop(columns = selected_features)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)



cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion:")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print("Exactitude:", accuracy)

precision = precision_score(y_test, y_pred)
print("Précision:", precision)

recall = recall_score(y_test, y_pred)
print("Rappel:", recall)

auc_roc = roc_auc_score(y_test, y_pred)
print("AUC-ROC:", auc_roc)
