from ..utils.utils import load_dataset

#loading the housing dataset
df = load_dataset('housing.csv')

def get_numerical_features(df):
    """Identifying the numerical features in a dataset

    Args:
        df (pd.DataFrame): The dataset to be analyzed
    Returns:
        list: The list of numerical features
    """
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.to_list()
    return numerical_features

def get_categorical_features(df):
    """Identifying the categorical features in a dataset

    Args:
        df (pd.DataFrame): The dataset to be analyzed
    Returns:
        list: The list of categorical features
    """
    categorical_features = df.select_dtypes(include=['object']).columns.to_list()
    return categorical_features


numerical_features = get_numerical_features(df)
categorical_features = get_categorical_features(df)

print("The numerical features are : ", numerical_features)
print("The categorical_features are:", categorical_features)