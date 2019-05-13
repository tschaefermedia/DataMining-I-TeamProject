import functions
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

x_train, y_train = functions.get_data()

criterion = ['gini', 'entropy']
max_depth = list(range(1, 21, 1)) + list(range(20, 110, 10))
splitter = ['best', 'random']
min_samples_split = list(range(2, 16, 2))
max_features = ['auto', 'sqrt', 'log2', None]

params = dict(criterion=criterion, max_depth=max_depth, splitter=splitter, min_samples_split=min_samples_split,
              max_features=max_features)

clf = DecisionTreeClassifier()

clfCV = GridSearchCV(clf, params, cv=10, scoring='accuracy', verbose=10, n_jobs=-1)

clfCV.fit(x_train, y_train.values.ravel())

best_score = clfCV.best_score_

best_estimator = clfCV.best_estimator_

best_params = clfCV.best_params_

f = open("results/files/clf_tuning_results.txt", "w+")

f.write("Best score = {0} \n".format(best_score))
f.write("Best estimator = {0} \n".format(best_estimator))
f.write("Best params = {0} \n".format(best_params))

f.close()
