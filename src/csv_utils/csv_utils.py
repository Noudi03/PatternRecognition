import pandas as pd

from importlib import resources


def construct_absolute_path(file_name):
    """Constructing the absolute path to a file in the data folder using importlib.resources

    Args:
        file_name (str): The name of the file to access
    Returns:
        str: The absolute path to the file
    """
    # accessing the resource as a file on the file system
    with resources.path('src.data', file_name) as data_path:
        return str(data_path)


def load_dataset(file_name):
    """Loading the CSV file of our dataset into a pandas DataFrame

    Args:
        file_name (str): The name of the CSV file
    Returns:
        pandas.DataFrame: The loaded DataFrame
    """
    file_path = construct_absolute_path(file_name)
    return pd.read_csv(file_path)
