from sklearn.neighbors import KNeighborsClassifier
import functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd

X, y = functions.get_data_experimental()

# split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

# Create KNN classifier
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                           metric_params=None, n_jobs=None, n_neighbors=96, p=2,
                           weights='distance')
# Fit the classifier to the data
knn.fit(X_train, y_train.values.ravel())

y_pred = knn.predict(X_test)

f1 = f1_score(y_test, y_pred, average='binary')

print("F1 score is = {0}".format(f1))

f = open("results/files/knn_results.txt", "w+")
f.write("Final F1 score = {0} \n".format(f1))
f.close()

result = pd.DataFrame(y_pred, columns=["y_pred"])

result['y_test'] = y_test["Loan Status"].values

result.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

result = pd.concat([result, X_test], axis=1)

result.to_csv("results/datasets/knn_y_pred.csv")
