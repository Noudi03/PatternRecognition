from sklearn.preprocessing import OneHotEncoder
from ..utils.utils import load_dataset, get_categorical_features

df = load_dataset('housing.csv')

def one_hot_encode_data(df, categorical_features):
    """One-hot encodes the categorical features of a dataset

    Args:
        df (pd.Dataframe): The dataset to be one-hot encoded
        categorical_features (list): The list of categorical features in the dataset

    Returns:
        result(numpy.ndarray): The one-hot encoded dataset
        
        NOTE: The output is in the form of a binary matrix:
        - Each row represents a sample.
        - Each column represents a category of ocean proximity.
        - Possible categories: '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'.
        - Every 0 value represents the absence of a category.
        - Every 1 value represents the presence of a category, allowing only one category per row.
    """
    
    #initializing the one-hot encoder
    encoder = OneHotEncoder()
    
    #selecting the categorical features only
    categorical_df = df[categorical_features]
    
    #fitting and transforming the data
    ohe_df = encoder.fit_transform(categorical_df)
    
    print(f"\nThe dataset has been one-hot encoded:\n")
    #converting the result from a scipy.sparse._csr.csr_matrix to a numpy.ndarray
    result = ohe_df.toarray()

    return result

#getting the list of the categorical features
categorical_features = get_categorical_features(df)

#applying the one-hot encoding to the dataset
ohe_df = one_hot_encode_data(df, categorical_features)

#printing the data
print(ohe_df)