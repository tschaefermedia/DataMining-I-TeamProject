from sklearn.metrics import f1_score
import pandas as pd
import functions
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X, y = functions.get_data()

# split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

clf = DecisionTreeClassifier(criterion="entropy", max_depth=10, splitter="random", min_samples_split=10,
                             max_features=None)
clf.fit(X_train, y_train.values.ravel())

y_pred = clf.predict(X_test)

f1 = f1_score(y_test, y_pred, average='binary')

print("F1 score is = {0}".format(f1))

f = open("results/files/clf_results.txt", "w+")
f.write("Final F1 score = {0} \n".format(f1))
f.close()

result = pd.DataFrame(y_pred, columns=["y_pred"])

result['y_test'] = y_test["Loan Status"].values

result.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

result = pd.concat([result, X_test], axis=1)

result.to_csv("results/datasets/clf_y_pred.csv")
