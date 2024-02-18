import pandas as pd
from utils import construct_absolute_path

#loading the housing dataset
csv_file_path = construct_absolute_path('housing.csv')
df = pd.read_csv(csv_file_path)

#identifying the numerical and categorical features
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.to_list()
categorical_features = df.select_dtypes(include=['object']).columns.to_list()

#printing the results
print("The numerical features are : ", numerical_features)
print("The categorical_features are:", categorical_features)