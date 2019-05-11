import functions
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

X, y = functions.get_data()

# split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# param values
neighbors = range(1, 32, 2)
weight = ["uniform", "distance"]
pp = [1, 2]
algo = ["auto"]
metric = ["minkowski", "manhattan"]

# creating  grid instance
grid_params_nn = dict(n_neighbors=neighbors, weights=weight, p=pp, algorithm=algo, metric=metric)

# creating KNN instance
knn = KNeighborsClassifier()

neigh_ins = GridSearchCV(knn, grid_params_nn, cv=10, scoring='f1', verbose=10, n_jobs=-1)

neigh_ins.fit(X_train, y_train.values.ravel())

best_score = neigh_ins.best_score_

best_estimator = neigh_ins.best_estimator_

best_params = neigh_ins.best_params_

f = open("results/files/knn_tuning_results.txt", "w+")

f.write("Best score = {0} \n".format(best_score))
f.write("Best estimator = {0} \n".format(best_estimator))
f.write("Best params = {0} \n".format(best_params))

f.close()

means = neigh_ins.cv_results_['mean_test_score']
stds = neigh_ins.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, neigh_ins.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()
