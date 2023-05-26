import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import roc_auc_score



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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()

# Perform forward stepwise selection
selector = SequentialFeatureSelector(model, direction='forward')
selector.fit(X_train, y_train)

# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)
params = selector.get_params(deep=True)


selected_features = X_train.columns[selected_feature_indices]

# Print the selected features and params
print("Selected Features:", selected_features)
print(params)
