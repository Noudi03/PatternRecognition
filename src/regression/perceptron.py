import pandas as pd
import numpy as np


def prepare_data():
    '''
        INPUTS:
            nothing
        RETURNS:
            input_data(pd.Dataframe): A dataset containing the data used as inputs for the model.
            
            target_data(pd.Dataframe): A dataset containing the data used by the error functions to train the model.
    '''

    #*Note: The dataset is already preprocessed and the missing values are filled with the median of the respective column.
    #a linear regression function g(x)= wx + b
    #where: w is the weight vector
    #       x is the feature vector
    #       b is the bias term


    #?REMOVE THIS LATER ON,LOAD DATAFRAME FROM MAIN.PY?NOT CERTAIN YET

    #loading the housing dataset with the filled median values
    df = pd.read_csv('../data/housing_filled.csv')
    
    #separating target from input data
    target_data = df["median_house_value"].copy()
    df.drop("median_house_value",axis=1);

    return (df,target_data)


def perceptron(input_data,weight_data,threshold):
    '''
        INPUTS:
            input_data(pd.Dataframe): the input vector X for our model,containing values X1 through Xn.
            
            weight_data(pd.Dataframe): the weight vector W for our model,containing values W1 through Wn.
                        
            bias(float): the bias we add to the sum before classification.Must be negative.
        RETURNS:
            1 if above threshold,0 if its less or equal with the threshold. 
    '''
    
    
    value_sum = 0
    
    #X1*W1 + X2*W2 +X3*W3 + ...Xn*Wn
    for (input,weight) in zip(input_data,weight_data):
        value_sum += input*weight
    
    #adding the W0/bias parameter.
    value_sum += bias
    
    #classifying into our two categories
    if value_sum > 0:
        return 1
    else:
        return 0