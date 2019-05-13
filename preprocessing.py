import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns

# ############LOADING DATASET #########################

data_folder = "data"
dataset = "credit_train"
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

# thresh12-10000, thresh=13-99999, thresh=14-99k, thresh=15-89k, thresh=16-79k, thresh=17-36.5k
df = df.dropna(thresh=17)

plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.show()

# removing string data from column to make it float type
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


# adding CreditRatio as new feature
def mi(row):
    ai = row["Annual Income"]
    md = row["Monthly Debt"]

    mr = ai / 12 - md
    return mr


def cr(row):
    cla = row["Current Loan Amount"]
    ai = row["Annual Income"]
    return cla / ai


df['Credit Score'] = df['Credit Score'].apply(cs)

# #fill NaN values groupby more related column
df['Credit Score'].fillna(value=df.groupby('Home Ownership')['Credit Score'].transform('median'), inplace=True)
df['Annual Income'].fillna(value=df.groupby('Purpose')['Annual Income'].transform('median'), inplace=True)
df['Years in current job'].fillna(value=df.groupby('Home Ownership')['Years in current job'].transform('median'),
                                  inplace=True)
df['Months since last delinquent'].fillna(0, inplace=True)
df['Maximum Open Credit'].fillna(value=df.groupby('Home Ownership')['Maximum Open Credit'].transform('median'),
                                 inplace=True)
df['Bankruptcies'].fillna(0, inplace=True)
df['Tax Liens'].fillna(0, inplace=True)

df["Monthly Income"] = df.apply(lambda row: mi(row), axis=1)
df["Credit Ration per Year"]= df.apply(lambda row: cr(row), axis=1)

df.info()

# #################################### Checking no. of Loan payers / non-payers #############

df.head(20)

df_paid = df[df['Loan Status'] == 'Fully Paid']
df_non_paid = df[df['Loan Status'] == 'Charged Off']

print("Number of people who paid their loan fully: {}".format(df_paid.shape[0]))
print("Number of people who didn't paid their loan fully: {}".format(df_non_paid.shape[0]))

# dealing  with categorical values

df_preprocessed = df.copy()

# create a variable with the values of the target variable
# loan_target = df['Loan Status']

# remove it from the dataframe so it only contains the features that our model should use
df = df.drop(df[['Home Ownership', 'Purpose']], axis=1)

# encode the target variable into a numeric value
label_encoder = preprocessing.LabelEncoder()
df['Loan Status'] = df['Loan Status'].astype(np.str)
df['Loan Status'] = label_encoder.fit_transform(df['Loan Status'])

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

plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.show()