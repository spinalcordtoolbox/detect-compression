import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV



from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

path_to_csv = '/Users/etiennedufayet/Desktop/STAGE_3A/Compression_detection_zurich/dataset_zurich_metrics_4.csv'

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


model = LogisticRegression()

# Perform forward stepwise selection
selector = SequentialFeatureSelector(model, direction='backward', n_features_to_select = 'auto', tol=None)
selector.fit(X_train, y_train)

# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)
params = selector.get_params(deep=True)


selected_features = X_train.columns[selected_feature_indices]
#selected_features = ['MEAN(area)', 'MEAN(diameter_AP)', 'MEAN(solidity)', 'MEAN(orientation)', 'slice_number', 'Torsion']

# Print the selected features and params

print("Selected Features:", selected_features)


'''
# adjust columns with respect to feature selection
X_train_logistic = X_train.drop(columns=[col for col in X_train.columns if col not in selected_features])
X_test_logistic = X_test.drop(columns=[col for col in X_test.columns if col not in selected_features])


#print(X_train['slice_number'].describe())



model.fit(X_train_logistic, y_train)


probabilities = model.predict_proba(X_test_logistic)[:, 1]


threshold_list = np.linspace(0,1,100)
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
'''

y_pred = np.where(probabilities > 0.105, 1, 0)

cm = confusion_matrix(y_pred, y_test)

print(cm)


### --- conclusion of the test: torsion does not really affect the output of the logistic regression 

## the automatic selection does not take into account the torsion colum
## let's compare both models graphically, with and without torsion

model_auto = LogisticRegression()
model_manual = LogisticRegression()


# Perform forward stepwise selection
selector = SequentialFeatureSelector(model_auto, direction='backward', n_features_to_select = 'auto', tol=None)
selector.fit(X_train, y_train)

# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)
params = selector.get_params(deep=True)


selected_features_auto = X_train.columns[selected_feature_indices]
selected_features_manual = ['MEAN(area)', 'MEAN(diameter_AP)', 'MEAN(solidity)', 'MEAN(orientation)', 'slice_number', 'Torsion']



# adjust columns with respect to feature selection
X_train_auto = X_train.drop(columns=[col for col in X_train.columns if col not in selected_features_auto])
X_test_auto = X_test.drop(columns=[col for col in X_test.columns if col not in selected_features_auto])

X_train_manual = X_train.drop(columns=[col for col in X_train.columns if col not in selected_features_manual])
X_test_manual = X_test.drop(columns=[col for col in X_test.columns if col not in selected_features_manual])

model_auto.fit(X_train_auto, y_train)
model_manual.fit(X_train_manual, y_train)

probabilities_auto = model_auto.predict_proba(X_test_auto)[:, 1]
probabilities_manual = model_manual.predict_proba(X_test_manual)[:, 1]


threshold_list = np.linspace(0,1,100)
accuracy_list_auto = []
precision_list_auto = []
recall_list_auto = []
auc_roc_list_auto = []

accuracy_list_manual = []
precision_list_manual = []
recall_list_manual = []
auc_roc_list_manual = []

for threshold in threshold_list:    
    y_pred_auto = np.where(probabilities_auto > threshold, 1, 0)
    y_pred_manual = np.where(probabilities_manual > threshold, 1, 0)


    accuracy_auto = accuracy_score(y_test, y_pred_auto)
    accuracy_list_auto.append(accuracy_auto)
    precision_auto = precision_score(y_test, y_pred_auto)
    precision_list_auto.append(precision_auto)
    recall_auto = recall_score(y_test, y_pred_auto)
    recall_list_auto.append(recall_auto)
    auc_roc_auto = roc_auc_score(y_test, y_pred_auto)
    auc_roc_list_auto.append(auc_roc_auto)

    accuracy_manual = accuracy_score(y_test, y_pred_manual)
    accuracy_list_manual.append(accuracy_manual)
    precision_manual = precision_score(y_test, y_pred_manual)
    precision_list_manual.append(precision_manual)
    recall_manual = recall_score(y_test, y_pred_manual)
    recall_list_manual.append(recall_manual)
    auc_roc_manual = roc_auc_score(y_test, y_pred_manual)
    auc_roc_list_manual.append(auc_roc_manual)


plt.plot(threshold_list, accuracy_list_auto, label='accuracy_auto')
plt.plot(threshold_list, precision_list_auto, label='precision_auto')
plt.plot(threshold_list, recall_list_auto, label='recall_auto')
plt.plot(threshold_list, auc_roc_list_auto, label='auc_roc_auto')

plt.plot(threshold_list, accuracy_list_manual, label='accuracy_manual')
plt.plot(threshold_list, precision_list_manual, label='precision_manual')
plt.plot(threshold_list, recall_list_manual, label='recall_manual')
plt.plot(threshold_list, auc_roc_list_manual, label='auc_roc_manual')

plt.legend()
plt.show()
'''


model = SVC()


# Perform forward stepwise selection
selector = SequentialFeatureSelector(model, direction='backward', n_features_to_select = 'auto', tol=None)
selector.fit(X_train, y_train)

# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)
params = selector.get_params(deep=True)


selected_features = X_train.columns[selected_feature_indices]
#selected_features = ['MEAN(area)', 'MEAN(diameter_AP)', 'MEAN(solidity)', 'MEAN(orientation)', 'slice_number', 'Torsion']

# Print the selected features and params

print("Selected Features:", selected_features)



# adjust columns with respect to feature selection
X_train_svm = X_train.drop(columns=[col for col in X_train.columns if col not in selected_features])
X_test_svm = X_test.drop(columns=[col for col in X_test.columns if col not in selected_features])


#print(X_train['slice_number'].describe())

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [0.1, 1, 10],
    'degree': [2, 3, 4]
}

grid_search = GridSearchCV(model, param_grid, cv=5)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

print(best_params)

'''
model.fit(X_train_svm, y_train)


probabilities = model.predict_proba(X_test_svm)[:, 1]


threshold_list = np.linspace(0,1,100)
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


plt.plot(threshold_list, accuracy_list, label='accuracy_svm')
plt.plot(threshold_list, precision_list, label='precision')
plt.plot(threshold_list, recall_list, label='recall')
plt.plot(threshold_list, auc_roc_list, label='auc_roc')


plt.legend()
plt.show()
'''