import pandas as pd
from sklearn import preprocessing


def get_data(standard=True, minmax=False):
    data_processed = pd.read_csv("data/credit_train_processed.csv", index_col=0)

    x_train, y_train = data_processed.drop(columns='Loan Status'), data_processed['Loan Status']

    min_max_scaler = preprocessing.MinMaxScaler()
    standard_scaler = preprocessing.StandardScaler()

    x_train['Term'] = x_train['Term'].astype('float64')
    x_train_standard = pd.DataFrame(standard_scaler.fit_transform(x_train), columns=x_train.columns)
    x_train_minmax = pd.DataFrame(min_max_scaler.fit_transform(x_train), columns=x_train.columns)

    y_train = pd.DataFrame(y_train)

    if standard:
        return x_train_standard, y_train
    elif minmax:
        return x_train_minmax, y_train
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
