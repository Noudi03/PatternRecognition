from sklearn.preprocessing import OneHotEncoder
from .data_type_identifier import get_categorical_features
import pandas as pd

def one_hot_encode_data(df):
    """One-hot encodes the categorical features of a dataset

    Args:
        df (pd.Dataframe): The dataset to be one-hot encoded

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
    categorical_df = df[get_categorical_features(df)]
    
    #fitting and transforming the data
    ohe_df = encoder.fit_transform(categorical_df)
    
    #converting the result from a scipy.sparse._csr.csr_matrix to a numpy.ndarray
    array_ohe = ohe_df.toarray()
    encoded_columns = encoder.get_feature_names_out(['ocean_proximity'])
    encoded_dataset = pd.DataFrame(array_ohe, columns=encoded_columns, dtype=int)
    
    print(f"\nThe categorical features of the dataset have been one-hot encoded:\n")
    
    return encoded_dataset

def append_categorical_data(df, ohe_df, output_csv_path):
    # Concatenate the one-hot encoded DataFrame with the original DataFrame
    final_df = pd.concat([df, ohe_df], axis=1)
    # Save the final DataFrame to a CSV file
    final_df.to_csv(output_csv_path, index=False)