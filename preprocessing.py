import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns

# ############LOADING DATASET #########################

data_folder = "data"
dataset = "credit_train"
file_type = ".csv"

df = pd.read_csv(data_folder+"/"+dataset+file_type)

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

df_final.to_csv(data_folder+"/"+dataset+"_processed"+file_type)