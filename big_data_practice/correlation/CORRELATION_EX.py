import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_iris, load_boston

iris = load_iris()
iris_data = iris.data
iris_feature_names = iris.feature_names
iris_df = pd.DataFrame(iris_data, columns=iris_feature_names)
iris_corr_matrix = iris_df.corr()
print("Correlation Matrix for Iris dataset:")
print(iris_corr_matrix)
print()
boston = load_boston()
boston_data = boston.data
boston_feature_names = boston.feature_names
boston_df = pd.DataFrame(boston_data, columns=boston_feature_names)
boston_corr_matrix = boston_df.corr()
print("Correlation Matrix for Boston Housing dataset:")
print(boston_corr_matrix)
print()
crim_corr = boston_df['CRIM'].corr(boston_df['TAX'])
print("Correlation between 'CRIM' and 'TAX' in Boston Housing dataset:", crim_corr)
