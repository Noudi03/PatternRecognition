import pandas as pd

#loading the dataset
df = pd.read_csv('housing.csv')

#identifying the numerical and categorical features
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.to_list()
categorical_features = df.select_dtypes(include=['object']).columns.to_list()

#printing the results
print("The numerical features are : ", numerical_features)
print("The categorical_features are:", categorical_features)

