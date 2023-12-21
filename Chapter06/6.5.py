import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV

datadir = "data"
dataset = "multiclass"
train = shuffle(pd.read_csv("data/ dataset/train.csv"))
test = shuffle(pd.read_csv("data/dataset/test.csv"))
train.isnull().values.any()
test.isnull().values.any()
train_outcome = pd.crosstab(index=train["Activity"],  # Makea crosstab
                            columns="count")  # Name the count column...
train_outcome
X_train = pd.DataFrame(train.drop(['Activity', 'subject '], axis=1))
Y_train_label = train.Activity.values.astype(object)
x_test = pd.DataFrame(test.drop(['Activity ', 'subject '], axis=1))
Y_test_label = test.Activity.values.astype(object)
from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()
encoder.fit(Y_train_label)
Y_train = encoder.transform(Y_train_label)
encoder.fit(Y_test_label)
Y_test = encoder.transform(Y_test_label)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(x_test)
params_grid = [{'kernel': ['rbf '], ' gamma ': [1e-3, 1e-4],
                'C': [1, 10, 100, 1000]},
               {'kernel': [' linear '], 'C': [1, 10, 100, 1000]}]
svm_model = GridSearchCV(SVC(), params_grid, cv=5)
svm_model.fit(X_train_scaled, Y_train)
print('Best score for training data : ', svm_model.best_score_, " \n ")
print('Best C: ', svm_model.best_estimator_.C, " \n")
print('Best Kernel : ', svm_model.best_estimator_.kernel, " \n")
print('Best Gamma : ', svm_model.best_estimator_.gamma, " \n ")
final_model = svm_model.best_estimator_
print("Training set score for SVM:%f" % final_model.score(X_train_scaled, Y_train))
print("Testing set score for SVM: %f" % final_model.score(x_test_scaled, Y_test))
