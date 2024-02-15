import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#loading the housing dataset
df = pd.read_csv('housing.csv')

#selecting the categorical feature 
categorical_df = df[['ocean_proximity']]

#initialization of the OneHotEncoder
encoder = OneHotEncoder()

#fitting and transforming the data
ohe_df = encoder.fit_transform(categorical_df)


"""
*NOTE:the output is in the form of a binary matrix,where each row represents a sample and each column represents a category of the ocean proximity.
*the possible categories are: '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'
*every 0 value in the matrix represents the absence of a category, while every 1 value represents the presence of a category, 
*thus only one category can have the value of 1 in each row.
"""
#printing the one hot encoded data
print(ohe_df.toarray())
