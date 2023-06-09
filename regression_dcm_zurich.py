import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

path_to_csv = '/Users/etiennedufayet/Desktop/STAGE_3A/Compression_detection_zurich/dataset_zurich_metrics.csv'

df = pd.read_csv(path_to_csv)

#print(df.info)

# data cleaning 
df = df.drop(columns=['DistancePMJ', 'VertLevel'])

df = df.dropna(axis=0)

# divide the dataset 
y = df['is_compressed']
X = df.drop(columns='is_compressed')

# reverse sigmoid: transform the non linear regression into a linear regression

# Add a constant column for the intercept term
X['intercept'] = 1

# Create an instance of the StandardScaler
scaler = StandardScaler()


# Apply the scaling transformation to both the training and test data
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2   , random_state=42, shuffle=True)

'''
print(X_train, y_train)
nombre_de_comp_train = (y_train == 1).sum()
nombre_de_comp_test = (y_test == 1).sum()
print(nombre_de_comp_train, nombre_de_comp_test)
'''

## scale the data 


model = LogisticRegression()

# Perform forward stepwise selection
selector = SequentialFeatureSelector(model, direction='backward', n_features_to_select = 7, tol=None)
selector.fit(X_train, y_train)

# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)
params = selector.get_params(deep=True)


selected_features = X_train.columns[selected_feature_indices]


# Print the selected features and params

#print("Selected Features:", selected_features)



# adjust columns with respect to feature selection
X_train = X_train.drop(columns=[col for col in X_train.columns if col not in selected_features])
X_test = X_test.drop(columns=[col for col in X_test.columns if col not in selected_features])


#print(X_train['slice_number'].describe())



model.fit(X_train, y_train)


print(X_test, y_test)


## get a test-dataset with only compressions 
X_compressed_test = X_test[y_test == 1]
Y_compressed_test = y_test[y_test == 1]

X_uncompressed_test = X_test[y_test == 0]
Y_uncompressed_test = y_test[y_test == 0]


probabilities_compressed = model.predict_proba(X_compressed_test)
probabilities_uncompressed = model.predict_proba(X_uncompressed_test)

probabilities_compressed = pd.DataFrame(probabilities_compressed)
probabilities_uncompressed = pd.DataFrame(probabilities_uncompressed)


#print(probabilities_compressed.describe())
#print(probabilities_uncompressed.describe())

probabilities = model.predict_proba(X_test)[:, 1]

'''
threshold_list = np.linspace(0,1,100)
cm_list = []
accuracy_list = []
precision_list = []
recall_list = []
auc_roc_list = []

for threshold in threshold_list:    
    y_pred = np.where(probabilities > threshold, 1, 0)


    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    precision = precision_score(y_test, y_pred)
    precision_list.append(precision)
    recall = recall_score(y_test, y_pred)
    recall_list.append(recall)
    auc_roc = roc_auc_score(y_test, y_pred)
    auc_roc_list.append(auc_roc)


plt.plot(threshold_list, accuracy_list, label='accuracy')
plt.plot(threshold_list, precision_list, label='precision')
plt.plot(threshold_list, recall_list, label='recall')
plt.plot(threshold_list, auc_roc_list, label='auc_roc')


plt.legend()
plt.show()

'''

y_pred = np.where(probabilities > 0.07, 1, 0)

cm = confusion_matrix(y_pred, y_test)

print(cm)