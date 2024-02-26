import pandas as pd
from preprocessing import get_numerical_features
from utils import construct_absolute_path


def check_empty_fields(path):
    """Checking for any empty numerical fields in a CSV file

    Args:
        path (str): Path to the CSV file
    Returns:
        None
    """

    df = pd.read_csv(path)
    numerical_df = df[get_numerical_features(df)]

    # checking for any missing values in the numerical dataset
    empty_fields = numerical_df.isnull().any().any()

# if there are any missing values, fill them with the median value of that column
    if empty_fields:

        # finding the number of empty numerical fields
        empty_field_count = numerical_df.isnull().sum().sum()

        # printing the number of empty fields found
        print(f"\n{empty_field_count} empty fields found in the CSV file.")

        # saving the filled dataset to a new CSV file in the data folder
        csv_file_path_filled = construct_absolute_path('housing_filled.csv')

        # filling the missing values with the median of the column
        fill_empty_fields(df, csv_file_path_filled)

    else:
        print("No empty fields found in the CSV file.\n")


def fill_empty_fields(df, path):
    """Filling filled_df, 'filled_df, 'filled_df, 'filledpty numerical fields in the dataset with the median of the column

    Args:
        df (pd.DataFrame): The dataset to be filled.
        path (str): Path to the CSV file where the filled dataset will be saved
    Returns:
        None
    """
    # creating a copy of the dataset
    df_filled = df.copy()
    numerical_df = df_filled[get_numerical_features(df)]

    # filling the empty fields of the dataset with the median of that fields column
    df_filled.fillna(numerical_df.median(), inplace=True)

    # saving the filled dataset to a new CSV file
    df_filled.to_csv(path, index=False)

    print("Missing values after filling empty fields",
          df_filled.isnull().sum().sum())
