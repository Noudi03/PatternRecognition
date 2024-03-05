import pandas as pd
import numpy as np


def split_dataframe_into_folds(df, k=10):
    ''' Splitting the dataframe into k folds for cross validation, while taking into account the distribution of the target variable.
        Args:
            df(pd.Dataframe): the dataframe to split.

            k(int): number of folds to split the dataframe into.
        Returns:
            fold_list(list[0...k](pd.Dataframe)): the separated folds to be used for cross validation.
    '''
    # selecting the target column
    target_column = "median_house_value"

    df_copy = df.copy()

    # dividing up the data into equal sized bins based on the percentiles of the target columns distribution
    df_copy['bin'] = pd.qcut(df_copy[target_column], q=k, labels=False)

    # shuffling the dataframe
    shuffled_df = df_copy.sample(frac=1).reset_index(drop=True)

    fold_list = []

    for bin_label in range(k):
        # creating an empty DataFrame for the current fold
        current_fold = pd.DataFrame()

        # sampling proportionally from each bin to construct the fold
        for bin_label, group in shuffled_df.groupby('bin'):
            # calculating the number of samples to take from this bin
            n_samples = int(np.rint(1/k * len(group)))
            sampled_group = group.sample(n_samples)
            # adding the sampled rows to the current fold
            current_fold = pd.concat([current_fold, sampled_group])

        # shuffling the constructed fold to mix rows from different bins and reset the index
        current_fold = current_fold.sample(frac=1).reset_index(drop=True)

        # adding the prepared fold to the list of folds
        fold_list.append(current_fold)

    # removing the bin column from each fold and the original dataframe
    for fold in fold_list:
        fold.drop(columns=['bin'], inplace=True)

    return fold_list
