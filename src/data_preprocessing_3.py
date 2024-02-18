import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from utils import construct_absolute_path

#loading the housing dataset
csv_file_path = construct_absolute_path('housing.csv')
df = pd.read_csv(csv_file_path)

#selecting the categorical feature 
categorical_df = df[['ocean_proximity']]

#initialization of the OneHotEncoder
encoder = OneHotEncoder()

#fitting and transforming the data
ohe_df = encoder.fit_transform(categorical_df)


"""
NOTE: The output is in the form of a binary matrix:
- Each row represents a sample.
- Each column represents a category of ocean proximity.
- Possible categories: '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'.
- Every 0 value represents the absence of a category.
- Every 1 value represents the presence of a category, allowing only one category per row.
"""

#printing the one hot encoded data
print(ohe_df.toarray())