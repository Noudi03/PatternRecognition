import pandas as pd
import numpy as np


def split_dataframe_into_folds(df,k = 10):
    '''
        INPUTS:
            df(pd.Dataframe): the dataframe to split.

            k(int): number of folds to split the dataframe into.
        RETURNS:
            fold_list(list[0...k](pd.Dataframe)): the separated folds to be used for cross validation.
    '''
    
    #finding how many entries each fold out of the k to be created shall contain.
    entry_num = len(df)//k

    shuffled_df = df.sample(frac = 1)

    fold_list = [shuffled_df[i:i+entry_num] for i in range(0,len(shuffled_df),entry_num)]
    
    return fold_list

