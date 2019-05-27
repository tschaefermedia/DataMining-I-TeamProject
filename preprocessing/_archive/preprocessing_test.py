import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns

# ############LOADING DATASET #########################

data_folder = "data"
dataset = "credit_test"
file_type = ".csv"

df = pd.read_csv(data_folder + "/" + dataset + file_type)

# removing unwanted features like LOanID and CustomerID
df = df.drop(df[['Loan ID', 'Customer ID']], axis=1)

# Drop record if it does not have at least 'n' values that are **not** NaN out of 17 columns
# finding val of 'n', we will see how many records are dropped at each threshlod val
for i in range(1, 18):
    mod_df = df.dropna(thresh=i)
    total_rows = mod_df.shape[0]
    print('With threshold value {0} the no. of records are {1}'.format(i, total_rows))

# for test 16 is the best value
df = df.dropna(thresh=16)

plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.show()

# # removing string data from column to make it float type
mapping_dict = {'8 years': 8, '10+ years': 10, '3 years': 3, '5 years': 5, '< 1 year': 0, '2 years': 2,
                '4 years': 4, '9 years': 9, '7 years': 7, '1 year': 1, '6 years': 6, 'n/a': np.nan}


def f(x):
    try:
        return mapping_dict[x]
    except:
        return x


df['Years in current job'] = df['Years in current job'].apply(f).astype(np.float64)


# Finding the median value in the respective columns
def cs(i):
    if i > 1000:
        i = i / 10
        return (i)
    else:
        return (i)


df['Credit Score'] = df['Credit Score'].apply(cs)

df['Credit Score'].fillna(value=df.groupby('Home Ownership')['Credit Score'].transform('median'), inplace=True)
df['Annual Income'].fillna(value=df.groupby('Purpose')['Annual Income'].transform('median'), inplace=True)
df['Years in current job'].fillna(value=df.groupby('Home Ownership')['Years in current job'].transform('median'), inplace=True)
df['Months since last delinquent'].fillna(0,inplace=True)
df['Maximum Open Credit'].fillna(value=df.groupby('Home Ownership')['Maximum Open Credit'].transform('median'),inplace=True)
df['Bankruptcies'].fillna(0,inplace=True)
df['Tax Liens'].fillna(0,inplace=True)

df.info()

# dealing  with categorical values

df_preprocessed = df.copy()

# create a variable with the values of the target variable
# loan_target = df['Loan Status']

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
encoded.info()

df.head(20)
# Run this cell only if you want to add the One Hot Encoded values for ['Home Ownership', 'Purpose'] into training data
# merging categorical encoded dataframe with the main dataframe and deleting unwanted features
df_final = df.reset_index(drop=True).merge(encoded.reset_index(drop=True), left_index=True, right_index=True)

df_final.to_csv(data_folder + "/" + dataset + "_processed" + file_type)
