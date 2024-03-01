from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from visualization import plot_histogram, plot_categorical, plot_variable_pairs
from csv_utils import load_dataset, construct_absolute_path
from preprocessing import get_numerical_features, get_categorical_features, check_empty_fields, ohe_data, append_data, scale_data
from regression import mlp_regression, perceptron_algorithm, least_squares_algorithm


def main():
    # loading the housing dataset
    df = load_dataset('housing.csv')

    # calling the functions to get the numerical and categorical features
    numerical_features = get_numerical_features(df)
    categorical_features = get_categorical_features(df)

    # displaying the results
    print("\nThe numerical features are : ", numerical_features)
    print("The categorical_features are:", categorical_features)

    # checking for empty fields in housing.csv
    csv_file_path = construct_absolute_path('housing.csv')
    check_empty_fields(csv_file_path)

    # checking for empty fields in housing_filled.csv
    csv_file_path_filled = construct_absolute_path('housing_filled.csv')
    check_empty_fields(csv_file_path_filled)

    filled_df = load_dataset('housing_filled.csv')

    # scaling the numerical features of the dataset
    scaled_df_standard = scale_data(
        filled_df, StandardScaler(), "StandardScaler", check_std=False)
    scale_data(filled_df, MinMaxScaler(), "MinMaxScaler")
    scale_data(filled_df, RobustScaler(), "RobustScaler")

    # applying the one-hot encoding to the dataset
    ohe_df = ohe_data(filled_df)
    print(ohe_df)

    # getting the final file we are going to use for the regression algorithms
    csv_file_path = construct_absolute_path('housing_final.csv')
    append_data(scaled_df_standard, ohe_df, csv_file_path)

    # visualization
    plot_categorical(filled_df)
    plot_histogram(filled_df)

    vars_to_plot_2 = ['total_rooms', 'total_bedrooms']
    plot_variable_pairs(df, vars_to_plot_2)
    vars_to_plot_3 = ['total_rooms', 'total_bedrooms', 'median_income']
    plot_variable_pairs(df, vars_to_plot_3)
    vars_to_plot_4 = ['total_rooms', 'total_bedrooms',
                      'median_income', 'median_house_value']
    plot_variable_pairs(df, vars_to_plot_4)

    # regression
    df = load_dataset('housing_final.csv')
    perceptron_algorithm(df, k=10, learning_rate=0.1)
    least_squares_algorithm(df, num_folds=10)
    mlp_regression(df)


if __name__ == "__main__":
    main()
