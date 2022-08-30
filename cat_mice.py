import pandas as pd # for dataframe
import numpy as np # for complex mathematical operation
from sklearn.tree import DecisionTreeRegressor # Decision Tree
from sklearn.preprocessing import OrdinalEncoder # For encoding and decoding categorical data
from sklearn.model_selection import train_test_split # For training and testing Model

import sklearn
from sklearn.experimental import enable_iterative_imputer # multiple Imputation is on experimental phase therefore We need to enable
from sklearn.impute import IterativeImputer # Multivariate Imputation using chain reaction

original_dataset = 'databases/HOV.xlsx' # Path/to/Original/dataset
incomplete_dataset = 'databases/HOV_AE_5.xlsx' # Path/to/incomplete/dataset
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
enc.fit(X)
X = enc.transform(X)

# Performing Multivariate Imputation by chain equation Using Decision Tree Regression
lr = DecisionTreeRegressor()
imp = IterativeImputer(estimator=lr, verbose=2, max_iter=max_itteration, tol=tolerance, imputation_order='roman')
output_df = imp.fit_transform(X)

# Decoding Categorical data and converting to Pandas data Frame as we are getting data in array
output_df = pd.DataFrame(enc.inverse_transform(output_df))
total_column = len(output_df.columns)
output_df.columns = [str(x) for x in range(1, total_column + 1)]

# Calculating AE
compared_df = output_df.eq(true_df)
n = true_df.size
upper_valeu = n - compared_df.sum().sum()
ae = upper_valeu/n
print("AE: ", ae)
