import pandas as pd # for dataframe
import numpy as np # for complex mathematical operation
from sklearn.linear_model import LinearRegression # Linear Regression
from sklearn.model_selection import train_test_split # For training and testing Model

import sklearn
from sklearn.experimental import enable_iterative_imputer # multiple Imputation is on experimental phase therefore We need to enable
from sklearn.impute import IterativeImputer # Multivariate Imputation using chain reaction

original_dataset = 'databases/Iris.xlsx' # Path/to/Original/dataset
incomplete_dataset = 'databases/Iris_AE_1.xlsx' # Path/to/incomplete/dataset
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

# Performing Multivariate Imputation by chain equation Using Linear Regression
lr = LinearRegression()
imp = IterativeImputer(estimator=lr, verbose=2, max_iter=max_itteration, tol=tolerance, imputation_order='roman')
output_df = imp.fit_transform(X)
output_df = pd.DataFrame(output_df, columns = [str(x) for x in range(1, total_column + 1)])

# Calculating NRMS
subtracted_df = output_df.subtract(true_df)
upper_valeu = np.square(subtracted_df).sum().sum() ** 0.5
lower_value = np.square(true_df).sum().sum() ** 0.5
nrms = upper_valeu/lower_value
print('NRMS:', nrms)
