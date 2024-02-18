import pandas as pd
from utils import construct_absolute_path

def check_empty_fields(path):
    """Checking for any empty numerical fields in a CSV file and filling them with the median of the column if any are found

    Args:
        path (str): Path to the CSV file
    Returns:
        None
    """
    
    df = pd.read_csv(path)
    numerical_df =  df.select_dtypes(include=['float64', 'int64'])

    #checking for any missing values in the numerical dataset
    empty_fields = numerical_df.isnull().any().any()
    #finding the number of empty numerical fields if any
    empty_field_count = numerical_df.isnull().sum().sum()

#if there are any missing values, fill them with the median value of that column
    if empty_fields:
        
        #printing the number of empty fields found
        print(f"{empty_field_count} empty fields found in the CSV file.")
        
        #filling the missing values with the median of the column
        df.fillna(numerical_df.median(), inplace=True)
        
        #saving the filled dataset to a new CSV file in the data folder
        csv_file_path_filled = construct_absolute_path('housing_filled.csv')
        df.to_csv(csv_file_path_filled, index=False)
        
        #checking if the missing values have been filled
        print("Missing values after filling:", df.isnull().sum().sum())
        
    else:
        print("No empty fields found in the CSV file.")


# For housing.csv
csv_file_path = construct_absolute_path('housing.csv')
check_empty_fields(csv_file_path)

# For housing_filled.csv
csv_file_path_filled = construct_absolute_path('housing_filled.csv')
check_empty_fields(csv_file_path_filled)

