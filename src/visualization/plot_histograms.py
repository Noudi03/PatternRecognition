import matplotlib.pyplot as plt

from preprocessing import get_numerical_features, get_categorical_features


def plot_histogram(df):
    """Plotting a histogram for each numerical column in a dataset

    Args:
        dataset (pd.DataFrame): The dataset to be plotted
    Returns:
        None
    """
    # using the numerical features only of the dataset
    numerical_df = df[get_numerical_features(df)]
    # plotting the histogram for each one of the numerical columns
    numerical_df.hist(bins=70, figsize=(20, 15))
    plt.show()


def plot_categorical(df):
    """Plotting a bar chart for each value of the ocean_proximity column in the dataset

    Args:
        dataset (pd.DataFrame): The dataset to be plotted
    Returns:
        None
    """

    # calculating the frequency of each different possible value of the ocean proximity column
    ocean_proximity = df[get_categorical_features(df)].value_counts()
    # these 5 different values are the only possible ones, so a true histogram isn't the most visually appealing way to represent the data
    ocean_proximity.plot(kind='barh')

    # adding the title and labels
    plt.title('Distribution of Ocean Proximity in the dataset')
    plt.ylabel('Ocean Proximity')
    plt.xlabel('Frequency')
    plt.show()
