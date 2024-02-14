import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#loading the dataset
df = pd.read_csv('housing.csv')

#selecting the numerical features only (could also return the values from the previous snippet might fix it)
numerical_df = df[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']]

#using 3 different scalers to compare the results

#applying the zscore scaler
scaled_df_zscore = numerical_df.apply(zscore)
print(scaled_df_zscore)

#initialization of the standard scaler, fitting and transforming the data
scaler = StandardScaler()
scaled_df_standard = scaler.fit_transform(numerical_df)
print(scaled_df_standard)

#initialization of the MinMax scaler, fitting and transforming the data
scaler = MinMaxScaler()
scaled_df_MinMax = scaler.fit_transform(numerical_df)
print(scaled_df_MinMax)