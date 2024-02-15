import pandas as pd

def check_empty_fields(path):
    df = pd.read_csv(path)
    numerical_df =  df.select_dtypes(include=['float64', 'int64'])

    #checking for any missing values in the numerical dataset
    empty_fields = numerical_df.isnull().any().any()
    #finding the number of empty numerical fields if any
    empty_field_count = numerical_df.isnull().sum().sum()

#if there are any missing values, they are going to be filled with the mean value of that column
    if empty_fields:
        
        #printing the number of empty fields found
        print(f"{empty_field_count} empty fields found in the CSV file.")
        
        #filling the missing values with the mean of the column
        df.fillna(numerical_df.mean(), inplace=True)
        
        #saving the filled dataset to a new CSV file
        df.to_csv('housing_filled_fields.csv', index=False)
        
        #checking if the missing values have been filled
        print("Missing values after filling:", df.isnull().sum().sum())
        
    else:
        print("No empty fields found in the CSV file.")


check_empty_fields('housing.csv')
#double checking if the missing values have been filled
check_empty_fields('housing_filled_fields.csv')