from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from visualization import plot_histogram, plot_categorical, plot_variable_pairs
from csv_utils import load_dataset, construct_absolute_path
from preprocessing import get_numerical_features, get_categorical_features, check_empty_fields, ohe_data, append_data, scale_data
from regression import mlp_regression, perceptron_algorithm, least_squares_algorithm


def main():
    # loading the housing dataset
    df = load_dataset('housing.csv')

    # calling the functions to get the numerical and categorical features
    input("\nShow numerical and categorical features: ")
    numerical_features = get_numerical_features(df)
    categorical_features = get_categorical_features(df)

    # displaying the results
    print("\nThe numerical features are: ", numerical_features)
    print("The categorical_features are: ", categorical_features)

    # checking for empty fields in housing.csv
    input("\nCheck for empty fields: ")
    csv_file_path = construct_absolute_path('housing.csv')
    check_empty_fields(csv_file_path)

    # checking for empty fields in housing_filled.csv
    csv_file_path_filled = construct_absolute_path('housing_filled.csv')
    check_empty_fields(csv_file_path_filled)

    filled_df = load_dataset('housing_filled.csv')

    # scaling the numerical features of the dataset
    input("\nScale the dataset with Standard Scaler: \n")
    scaled_df_standard = scale_data(
        filled_df, StandardScaler(), "StandardScaler", check_std=False)
    input("\nScale the dataset with MinMax Scaler: ")
    scale_data(filled_df, MinMaxScaler(), "MinMaxScaler")
    input("\nScale the dataset with Robust Scaler: ")
    scale_data(filled_df, RobustScaler(), "RobustScaler")

    # applying the one-hot encoding to the dataset
    input("\nOne hot encode the categorical features: ")
    ohe_df = ohe_data(filled_df)
    print(ohe_df)

    # getting the final file we are going to use for the regression algorithms
    csv_file_path = construct_absolute_path('housing_final.csv')
    append_data(scaled_df_standard, ohe_df, csv_file_path)

    # visualization
    input("\nPlot the categorical data: ")
    plot_categorical(filled_df)
    input("\nPlot histograms: ")
    plot_histogram(filled_df)

    input("\nPlot pairs for 2 variables: ")
    vars_to_plot_2 = ['total_rooms', 'total_bedrooms']
    plot_variable_pairs(df, vars_to_plot_2)
    input("\nPlot pairs for 3 variables: ")
    vars_to_plot_3 = ['total_rooms', 'total_bedrooms', 'median_income']
    plot_variable_pairs(df, vars_to_plot_3)
    input("\nPlot pairs for 4 variables: ")
    vars_to_plot_4 = ['total_rooms', 'total_bedrooms',
                      'median_income', 'median_house_value']
    plot_variable_pairs(df, vars_to_plot_4)

    # regression
    df = load_dataset('housing_final.csv')
    input("\nRun perceptron:\n")
    perceptron_algorithm(df, k=10, learning_rate=0.1)
    input("\nRun least squares:\n ")
    least_squares_algorithm(df, num_folds=10)
    input("\nRun mlp regression:\n ")
    mlp_regression(df)


if __name__ == "__main__":
    main()
