import functions
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV

x_train, y_train = functions.get_data()
# param values
neighbors = range(1, 32, 2)
weight = ["uniform", "distance"]
pp = [1, 2]
algo = ["auto", "ball_tree", "kd_tree"]
metric = ["minkowski", "manhattan"]

grid_params_nn = dict(n_neighbors=neighbors, weights=weight, p=pp, algorithm=algo, metric=metric)
# creating  grid instance
knn = KNeighborsClassifier()

neigh_ins = RandomizedSearchCV(knn, grid_params_nn, cv=10, scoring='f1', n_iter=384, verbose=10, n_jobs=-1)

neigh_ins.fit(x_train, y_train.values.ravel())

best_score = neigh_ins.best_score_

best_estimator = neigh_ins.best_estimator_

best_params = neigh_ins.best_params_

f = open("results/files/knn_tuning_results.txt", "w+")

f.write("Best score = {0} \n".format(best_score))
f.write("Best estimator = {0} \n".format(best_estimator))
f.write("Best params = {0} \n".format(best_params))

f.close()
