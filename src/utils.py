import os

def construct_absolute_path(file_name):
    """Constructing the absolute path to a file in the data folder

    Args:
        file_name (str): The name of the file
    Returns:
        str: The absolute path to the file
    """
    data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
    return os.path.join(data_folder, file_name)