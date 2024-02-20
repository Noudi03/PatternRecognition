import pandas as pd
import numpy as np



def prepare_input_and_target_data():
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
    df = pd.read_csv('data\housing_filled.csv')
    #separating target from input data
    value_data = df["median_house_value"].copy()

    #setting threshold as the median of all values
    threshold = value_data.median()

    #populating the target_data vector for use in the model with 0 being given to data 
    #with median value below or equal the threshold an the opposite for those above
    target_data = np.where(value_data > threshold, 1, 0)
    df.drop("median_house_value",axis=1);

    return (df,target_data)


def initialize_weight_data(df_len):
    '''
        INPUTS:
            df_len(int): the number of rows in our dataframe.Can get by using the built-in len(df) function.

        RETURNS:
            weight_data(list): the initialized weight data for the first iteration of the model.
    '''
    



def perceptron(input_data,weight_data,threshold):
    '''
        INPUTS:
            input_data(pd.Dataframe): the input vector X for our model,containing values X1 through Xn.
            
            weight_data(pd.Dataframe): the weight vector W for our model,containing values W1 through Wn.
                        
            bias(float): the bias we add to the sum before classification.
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
    



