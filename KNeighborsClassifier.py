import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

data_processed = pd.read_csv("data/credit_train_processed.csv", index_col=0)

x_train, y_train = data_processed.drop(columns='Loan Status'), data_processed['Loan Status']

min_max_scaler = preprocessing.MinMaxScaler()
standard = StandardScaler()

x_train['Term'] = x_train['Term'].astype('float64')
x_train_standard = pd.DataFrame(standard.fit_transform(x_train), columns=x_train.columns)
x_train_minmax = pd.DataFrame(min_max_scaler.fit_transform(x_train), columns=x_train.columns)

y_train = pd.DataFrame(y_train)

# param values
neighbors = range(1, 32, 2)
weight = ["uniform", "distance"]
pp = [1, 2]
algo = ["auto", "ball_tree", "kd_tree", "brute"]
metric = ["minkowski", "manhattan", "chebyshev"]
# all param-val dictionary
# grid_params_lr = dict('C':[C_regularization], 'penalty':["l1","l2"], 'intercept_scaling':[intercept_scal_vals],
# 'max_iter':[max_iter_vals], 'solver' :["newton-cg", "llbfgs", "sag"])

grid_params_nn = dict(n_neighbors=neighbors, weights=weight, p=pp, algorithm=algo, metric=metric)
# creating  grid instance
# KNeighborsClassifier(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2)
knn = KNeighborsClassifier()
# neigh_grid=GridSearchCV(knn,grid_params_nn,cv=10)
neigh_ins = RandomizedSearchCV(knn, grid_params_nn, cv=10, scoring='f1', n_iter=768, verbose=10, n_jobs=-1)

neigh_ins.fit(x_train_standard, y_train.values.ravel())

best_score = neigh_ins.best_score_

best_estimator = neigh_ins.best_estimator_

best_params = neigh_ins.best_params_

f = open("knn_results.txt", "w+")

f.write("Best score = {0}".format(best_score))
f.write("Best estimator = {0}".format(best_estimator))
f.write("Best params = {0}".format(best_params))

f.close()
