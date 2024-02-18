from utils.utils import load_dataset, construct_absolute_path
from preprocessing.data_type_identifier import get_numerical_features, get_categorical_features
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from preprocessing.numerical_scaling import scale_data
from preprocessing.one_hot_encode import one_hot_encode_data
from preprocessing.fill_data import check_empty_fields
from visualization.plot_histograms import plot_histogram, plot_categorical
from visualization.plot_pairs import plot_variable_pairs
#loading the housing dataset
df = load_dataset('housing.csv')

#calling the functions to get the numerical and categorical features
numerical_features = get_numerical_features(df)
categorical_features = get_categorical_features(df)

#displaying the results
print("The numerical features are : ", numerical_features)
print("The categorical_features are:", categorical_features)

scaled_df_standard = scale_data(df, StandardScaler(), "StandardScaler", check_std=False)
scaled_df_MinMax = scale_data(df, MinMaxScaler(), "MinMaxScaler")
scaled_df_Robust = scale_data(df, RobustScaler(), "RobustScaler")

#applying the one-hot encoding to the dataset
ohe_df = one_hot_encode_data(df)

#printing the data
print(ohe_df)

#checking for empty fields in housing.csv
csv_file_path = construct_absolute_path('housing.csv')
check_empty_fields(csv_file_path)

#checking for empty fields in housing_filled.csv
csv_file_path_filled = construct_absolute_path('housing_filled.csv')
check_empty_fields(csv_file_path_filled)

filled_df = load_dataset('housing_filled.csv')
plot_categorical(filled_df)
plot_histogram(filled_df)

vars_to_plot_2 = ['total_rooms', 'total_bedrooms']
plot_variable_pairs(df, vars_to_plot_2)
vars_to_plot_3 = ['total_rooms', 'total_bedrooms', 'median_income']
plot_variable_pairs(df, vars_to_plot_3)
vars_to_plot_4 = ['total_rooms', 'total_bedrooms', 'median_income', 'median_house_value']
plot_variable_pairs(df, vars_to_plot_4)