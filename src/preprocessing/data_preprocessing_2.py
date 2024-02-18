import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from ..utils.utils import load_dataset,  get_numerical_features

#loading the housing dataset with missing values filled
df = load_dataset('housing_filled.csv')

#!using 4 different scalers to compare the results

def scale_data(df, scaler, scaler_name, check_std=True):
    """Scaling the numerical features of a dataset using the specified scaler
        Types supported: StandardScaler, MinMaxScaler, RobustScaler, zscore Scaler

    Args:
        df (pd.DataFrame): The dataset to be scaled
        scaler (sklearn.preprocessing._data.StandardScaler,
                sklearn.preprocessing._data.MinMaxScaler, 
                sklearn.preprocessing._data.RobustScaler
                scipy.stats.zscore Scaler): The scaler to be used
        check_std (bool, optional): Whether to check the standard deviation of the columns. Defaults to True.
    Returns:
        pd.DataFrame: The scaled dataset
        check_std (bool, optional): Whether to check the standard deviation of the columns. Defaults to True.
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

    if scaler_name == "zscore":
        scaled_df = numerical_df.apply(zscore)
    else:
        
        scaled_df = scaler.fit_transform(numerical_df) #returns a numpy.ndarray
        #so we need to convert it back to a pd.Dataframe for displaying uniformity
        scaled_df = pd.DataFrame(scaled_df, columns=numerical_df.columns)
    
    print(f"\nThe dataset has been scaled using {scaler_name}:\n")
    print(scaled_df)
    return scaled_df

scaled_df_zscore = scale_data(df, zscore, "zscore", check_std=False)
scaled_df_standard = scale_data(df, StandardScaler(), "StandardScaler")
scaled_df_MinMax = scale_data(df, MinMaxScaler(), "MinMaxScaler")
scaled_df_Robust = scale_data(df, RobustScaler(), "RobustScaler")