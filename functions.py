import pandas as pd
import numpy as np
from sklearn import preprocessing


def get_data(standard=True, minmax=False, normal=False, type=""):
    data_processed = pd.read_csv("../data/credit_train_processed"+type+".csv", index_col=0)

    x_train, y_train = data_processed.drop(columns='Loan Status'), data_processed['Loan Status']

    min_max_scaler = preprocessing.MinMaxScaler()
    standard_scaler = preprocessing.StandardScaler()

    if type == "" or type == "__removeoutliers":
        x_train['Term'] = x_train['Term'].astype('float64')
    x_train_standard = pd.DataFrame(standard_scaler.fit_transform(x_train), columns=x_train.columns)
    x_train_minmax = pd.DataFrame(min_max_scaler.fit_transform(x_train), columns=x_train.columns)

    y_train = pd.DataFrame(y_train)

    if standard:
        return x_train_standard, y_train
    elif minmax:
        return x_train_minmax, y_train
    elif normal: 
        return x_train, y_train
    else:
        return False


def get_data_experimental(standard=True, minmax=False):
    data_processed = pd.read_csv("data/credit_train_processed.csv", index_col=0)

    x_train, y_train = data_processed.drop(columns='Loan Status'), data_processed['Loan Status']

    min_max_scaler = preprocessing.MinMaxScaler()
    standard_scaler = preprocessing.StandardScaler()

    x_train['Term'] = x_train['Term'].astype('float64')
    x_train_standard = pd.DataFrame(standard_scaler.fit_transform(x_train), columns=x_train.columns)
    x_train_minmax = pd.DataFrame(min_max_scaler.fit_transform(x_train), columns=x_train.columns)

    y_train = pd.DataFrame(y_train)
    if standard:
        x = x_train_standard[["Credit Score", "Monthly Debt", "Bankruptcies",
                              "Number of Credit Problems", "Monthly Income", "Credit Ration per Year"]]
        print("Used columns: ")
        print(list(x.columns.values))
        return x, y_train
    elif minmax:
        return x_train_minmax, y_train
    else:
        return False


def f(x):
    # define already build functions
    # removing string data from column to make it float type
    mapping_dict = {'8 years': 8, '10+ years': 10, '3 years': 3, '5 years': 5, '< 1 year': 0, '2 years': 2,
                    '4 years': 4, '9 years': 9, '7 years': 7, '1 year': 1, '6 years': 6, 'n/a': np.nan}
    try:
        return mapping_dict[x]
    except:
        return x


# Finding the median value in the respective columns
def cs(i):
    if i > 1000:
        i = i / 10
        return i
    else:
        return i


# adding monthly income
def mi(row):
    ai = row["Annual Income"]
    md = row["Monthly Debt"]

    miv = ai / 12 - md
    return miv


# adding CreditRatio as new feature
def cr(row):
    cla = row["Current Loan Amount"]
    ai = row["Annual Income"]
    return cla / ai
