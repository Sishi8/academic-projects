import pandas as pd # for dataframe
import numpy as np # for complex mathematical operation
from sklearn.tree import DecisionTreeRegressor # Decision Tree
from sklearn.preprocessing import OrdinalEncoder # For encoding and decoding categorical data
from sklearn.model_selection import train_test_split # For training and testing Model

import sklearn
from sklearn.experimental import enable_iterative_imputer # multiple Imputation is on experimental phase therefore We need to enable
from sklearn.impute import IterativeImputer # Multivariate Imputation using chain reaction

list_cat_col = ['2', '3', '4'] # List of all the column names of Categorical data starting from 1 to n
original_dataset = 'databases/KDD.xlsx' # Path/to/Original/dataset
incomplete_dataset = 'databases/KDD_AE_10.xlsx' # Path/to/incomplete/dataset
max_itteration = 100 # Maximum number of itteration
tolerance = 0.01e-10 # Tolerance rate

# Getting Original Dataset and adding column names from 1 to n
true_df = pd.read_excel(original_dataset, header = None)
total_column = len(true_df.columns)
true_df.columns = [str(x) for x in range(1, total_column + 1)]

# Function to get incomplete Dataset and adding column names from 1 to n
def do_formality(file_location):
    given_df = pd.read_excel(file_location, header=None)
    total_column = len(given_df.columns)
    given_df.columns = [str(x) for x in range(1, total_column + 1)]
    return given_df

df = do_formality(incomplete_dataset)
X = df

# Encoding Categorical data
enc = OrdinalEncoder()
enc.fit(X[list_cat_col])
X[list_cat_col] = enc.transform(X[list_cat_col])

# Performing Multivariate Imputation by chain equation Using Decision Tree Regression
lr = DecisionTreeRegressor()
imp = IterativeImputer(estimator=lr, verbose=2, max_iter=max_itteration, tol=tolerance, imputation_order='roman')
output_df = imp.fit_transform(X)

# We get data in array therefore converting to Pandas data Frame
output_df = pd.DataFrame(output_df)

total_column = len(output_df.columns)
output_df.columns = [str(x) for x in range(1, total_column + 1)]

# Decoding the Categorical data
output_df[list_cat_col] = enc.inverse_transform(output_df[list_cat_col])

# Calculating AE
compared_df = output_df[list_cat_col].eq(true_df[list_cat_col])
n = true_df[list_cat_col].size
upper_value_ae = n - compared_df.sum().sum()
ae = upper_value_ae/n
print("AE: ", ae)

# Calculating NRMS
true_df_nrms = true_df.drop(list_cat_col, axis = 1)
output_df_nrms = output_df.drop(list_cat_col, axis = 1)
subtracted_df = output_df_nrms.subtract(true_df_nrms)
upper_value_nrms = np.square(subtracted_df).sum().sum() ** 0.5
lower_value_nrms = np.square(true_df_nrms).sum().sum() ** 0.5
nrms = upper_value_nrms/lower_value_nrms
print('nrms: ', nrms)
