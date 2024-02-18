import pandas as pd
import os

data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
csv_file_path = os.path.join(data_folder, 'housing.csv')

#loading the housing dataset
df = pd.read_csv(csv_file_path)

#identifying the numerical and categorical features
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.to_list()
categorical_features = df.select_dtypes(include=['object']).columns.to_list()

#printing the results
print("The numerical features are : ", numerical_features)
print("The categorical_features are:", categorical_features)