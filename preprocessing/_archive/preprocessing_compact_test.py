import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns

# Compact version - only data, no plots or infos

# ############LOADING DATASET #########################

data_folder = "data"
dataset = "credit_test"
file_type = ".csv"

df = pd.read_csv(data_folder + "/" + dataset + file_type)

# removing unwanted features like LOanID and CustomerID
df = df.drop(df[['Loan ID', 'Customer ID']], axis=1)

# for test 16 is the best value
df = df.dropna(thresh=16)

# removing string data from column to make it float type
df = df.replace({'Years in current job': '[A-Za-z+<>]'}, ' ', regex=True)

# changing object value to float
df['Years in current job'] = df['Years in current job'].astype(np.float64)

# Finding the median value in the rspective columns

credit_score_median = df['Credit Score'].median()
Annual_Income_median = df['Annual Income'].median()
Years_in_current_job_median = df['Years in current job'].median()

# fill NaN values
df['Credit Score'].fillna(credit_score_median, inplace=True)
df['Annual Income'].fillna(Annual_Income_median, inplace=True)
df['Years in current job'].fillna(Years_in_current_job_median, inplace=True)
df['Months since last delinquent'].fillna(0, inplace=True)
df['Maximum Open Credit'].fillna(df['Maximum Open Credit'].median(), inplace=True)
df['Bankruptcies'].fillna(0, inplace=True)
df['Tax Liens'].fillna(0, inplace=True)

# dealing  with categorical values

df_preprocessed = df.copy()

# remove it from the dataframe so it only contains the features that our model should use
df = df.drop(df[['Home Ownership', 'Purpose']], axis=1)

label_encoder1 = preprocessing.LabelEncoder()
df['Term'] = label_encoder1.fit_transform(df['Term'])

# encode features
encoder = preprocessing.OneHotEncoder()
encoded = pd.DataFrame(encoder.fit_transform(df_preprocessed[['Home Ownership', 'Purpose']]).toarray(),
                       columns=encoder.get_feature_names(['Home Ownership', 'Purpose']))
encoded.info()

# ################### Run this cell only if you want to add One hot encoding values for columns ['Home Ownership', 'Purpose']

# encode features
encoder = preprocessing.OneHotEncoder()
encoded = pd.DataFrame(encoder.fit_transform(df_preprocessed[['Home Ownership', 'Purpose']]).toarray(),
                       columns=encoder.get_feature_names(['Home Ownership', 'Purpose']))

# Run this cell only if you want to add the One Hot Encoded values for ['Home Ownership', 'Purpose'] into training data
# merging categorical encoded dataframe with the main dataframe and deleting unwanted features
df_final = df.reset_index(drop=True).merge(encoded.reset_index(drop=True), left_index=True, right_index=True)

df_final.to_csv(data_folder + "/" + dataset + "_processed" + file_type)
