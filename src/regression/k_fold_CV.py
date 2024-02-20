import pandas as pd
import numpy as np


def split_dataframe_into_folds(k,df):
    '''
        INPUTS:
            k(int): number of folds to split the dataframe into.

            df(pd.Dataframe): the dataframe to split.
        RETURNS:
            fold_list(list[0...k](pd.Dataframe)): the separated folds to be used for cross validation.
    '''
    
    #finding how many entries each fold out of the k to be created shall contain.
    entry_num = len(df)//10

    shuffled_df = df.sample(frac = 1)

    fold_list = [shuffled_df[i:i+entry_num] for i in range(0,len(shuffled_df),entry_num)]
    return fold_list
    #! REMOVE THIS SHIT BEFORE UPLOADING ONLY USE FOR TESTING THIS FUNCTION
    #for i in range(0,10):
        #print(fold_list[i])

#! REMOVE THIS SHIT BEFORE UPLOADING ONLY USE FOR TESTING THIS FUNCTION
#df = pd.read_csv('data\housing_filled.csv')
#split_dataframe_into_folds(10,df)   