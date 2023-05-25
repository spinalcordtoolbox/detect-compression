import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


path_to_csv = '/Users/etiennedufayet/Desktop/STAGE_3A/Compression_detection_zurich/dataset_zurich_metrics.csv'

df = pd.read_csv(path_to_csv)

y = df['is_compressed']
X = df.drop(columns='is_compressed', inplace=True)


for value in y:
    if value == 0: 
        value = 10**(-10)
    
    else: 
        value = 1 - 10**(-10)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X.info()