import os

import pandas


def construct_absolute_path(file_name):
    """Constructing the absolute path to a file in the data folder

    Args:
        file_name (str): The name of the file
    Returns:
        str: The absolute path to the file
    """
    data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    return os.path.join(data_folder, file_name)


def load_dataset(file_name):
    """Loading the CSV file of our dataset into a pandas DataFrame

    Args:
        file_name (str): The name of the CSV file
    Returns:
        pandas.DataFrame: The loaded DataFrame
    """
    file_path = construct_absolute_path(file_name)
    return pandas.read_csv(file_path)
