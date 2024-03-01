import numpy as np


def mean_square_error(prediction_data,target_data):
    '''
        INPUTS:
            prediction_data(list(int)): a list of 0s and 1s representing our model's predictions

            target_data(list(int)): a list of 0s and 1s showing us what the model should predict
        RESULTS:
            average_cost(float): the average cost of our model 

            loss(list(float)): the individual loss values for each piece of prediction and target data
    '''
    cost_sum = 0
    #getting the number of entries to use to get the average later
    entry_num = len(prediction_data)
    loss = []
    for index,prediction in enumerate(prediction_data):
        cost_sum += (target_data[index] - prediction)**2
        loss.append((target_data[index] - prediction)**2)

    average = cost_sum/entry_num 
    return (average)

def mean_absolute_error(prediction_data,target_data):
    '''
        INPUTS:
            prediction_data(list(int)): a list of 0s and 1s representing our model's predictions

            target_data(list(int)): a list of 0s and 1s showing us what the model should predict

        RESULTS:
            mean_cost(float): the average cost of our model 
    '''
    cost_sum = 0
    #getting the number of entries to use to get the average later
    entry_num = len(prediction_data)

    loss = []

    for index,prediction in enumerate(prediction_data):
        cost_sum += abs(target_data[index] - prediction)
        loss.append(target_data[index] - prediction)

    average = cost_sum/entry_num 
    return (average)


def calculate_square_error_matrix(target_matrix,prediction_matrix):
    '''
        INPUTS:
            target_matrix(np.array): The target matrix Y consisting of the target data that we aim to achieve with our model

            prediction_matrix(np.array): The prediction matrix Y' consisting of the prediction data our model actually calculated
        RETURNS:
            E_matrix(np.array): The error matrix showing us the square loss of each individual prediction 
    '''
    #*FOR COST CALCULATIONS TO INITIALIZE THE E MATRIX WE WILL USE THE SQUARE LOSS ALGORITHM,WITH FORMULA BEING:
    #*E = (Y - Y')^2 
    E_matrix = np.power(target_matrix - prediction_matrix,2)
    return(E_matrix)



def calculate_absolute_error_matrix(target_matrix,prediction_matrix):
    '''
        INPUTS:
            target_matrix(np.array): The target matrix Y consisting of the target data that we aim to achieve with our model

            prediction_matrix(np.array): The prediction matrix Y' consisting of the prediction data our model actually calculated
        RETURNS:
            E_matrix(np.array): The error matrix showing us the absolute loss of each individual prediction 
    '''
    #*FOR COST CALCULATIONS TO INITIALIZE THE E MATRIX WE WILL USE THE SQUARE LOSS ALGORITHM,WITH FORMULA BEING:
    #*E = |Y - Y'| 
    E_matrix = abs(target_matrix - prediction_matrix)
    return(E_matrix)
