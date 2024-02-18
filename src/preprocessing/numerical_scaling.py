import pandas as pd
from .data_type_identifier import get_numerical_features

#!using 3 different scalers to compare the results

def scale_data(df, scaler, scaler_name, check_std=True):
    """Scaling the numerical features of a dataset using the specified scaler
        Types supported: StandardScaler, MinMaxScaler, RobustScaler, zscore Scaler

    Args:
        df (pd.DataFrame): The dataset to be scaled
        scaler (sklearn.preprocessing._data.StandardScaler,
                sklearn.preprocessing._data.MinMaxScaler, 
                sklearn.preprocessing._data.RobustScaler): The scaler to be used
        check_std (bool, optional): Whether to check the standard deviation of the columns. Defaults to True.
    Returns:
        pd.DataFrame: The scaled dataset
    """
    
    #selecting the numerical features only
    numerical_df = df[get_numerical_features(df)]
    
    #checking to see what the standard deviation of each column is
    #there could be a possible division by 0 if the standard deviation is 0
    #in this case this isn't a problem but i'm leaving this here anyway
    if not check_std:
        for column in numerical_df.columns:
            column_std = numerical_df[column].std()
            print(f"The standard deviation of {column} is: {column_std}")
            if column_std == 0:
                print(f"The standard deviation of {column} is 0, so it is not possible to apply the scaler to this column")
                numerical_df = numerical_df.drop(column, axis=1)

    
    scaled_df = scaler.fit_transform(numerical_df) #returns a numpy.ndarray
    #so we need to convert it back to a pd.Dataframe for displaying uniformity
    scaled_df = pd.DataFrame(scaled_df, columns=numerical_df.columns)
    
    print(f"\nThe dataset has been scaled using {scaler_name}:\n")
    print(scaled_df)
    return scaled_df