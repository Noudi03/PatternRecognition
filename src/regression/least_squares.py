import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from numpy.linalg import inv

from loss_functions import calculate_absolute_error_matrix,calculate_square_error_matrix

def calculate_slope_coefficient_matrix(x_matrix,y_matrix):
    '''
        INPUTS:
            x_matrix(np.array): The parameter matrix X,with size nXk where n is the number of samples in our dataset and k is the number of parameters we give to our model
        
            y_matrix(np.array): The target matrix Y,with size nX1 where n is the number of samples in our dataset
        RETURNS:
            B_matrix(np.array): The slope coefficient matrix B,calculated using the formula B = (inverse((transpose(X)*X))) * transpose(X)*Y
    '''
    #* THE FORMULA TO CALCULATE THE B MATRIX IS:
    #* B = (inverse((transpose(X)*X))) * transpose(X)*Y
    x_transpose = x_matrix.transpose()
    #* temp_A = (transpose(X)*X)^-1)
    temp_A = inv(np.matmul(x_transpose,x_matrix)) 
    #* temp_B = transpose(X)*Y
    temp_B = np.matmul(x_transpose,y_matrix)
    B_matrix = np.matmul(temp_A,temp_B)
    return(B_matrix)


def least_squares(input_data,num_folds = 10):
    '''
        INPUTS:
            input_data(pd.Dataframe): The dataframe we plan to feed our model scaled appropriately and with categorical values encoded with one hot encoding beforehand.

            num_folds(int): The number of folds we shall use for K-cross validation.With K being the number of folds.Default is set to 10 for 10-cross validation.
    '''
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)


    #splitting the data into features and target
    X = input_data.drop('median_house_value', axis=1)
    y = input_data['median_house_value']


    #*initializing the score variables
    training_mse_scores = 0
    training_mae_scores = 0
    testing_mse_scores = 0
    testing_mae_scores = 0

    #*train_index refers to the indeces of our current train folds,while the test_index to the index of our current test fold
    #*We gonna initialize 2 different sets of numpy arrays here.One being X,representing the input and the other being Y,representing the target values.
    fold_index = 1
    for train_index,test_index in kf.split(X):
        

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train_array = X_train.to_numpy()
        Y_train_array = y_train.to_numpy()


        #*MATHEMATICAL NOTES:
        #*THERE ARE 4 MATRICES OF IMPORTANCE:
        #* I)Y representing our target/dependent values,with size n*1
        #* II)X representing our inputs/independent values with size n*k
        #* III)B representing our sloap coefficients with size k*1
        #* IV)E representing our errors with size n*1
        B = calculate_slope_coefficient_matrix(X_train_array,Y_train_array)
        #!print(B.shape)
        #*CALCULATING THE PREDICTIONS:
        #*REMINDER THAT FOR PREDICTION Y' AND LINE THAT STARTS AT (0,0) OUR FORMULA IS Y' = TRANSPOSE(X)*B with B BEING THE SLOPE COEFFICIENTS OF OUR PARAMETERS IN THE X MATRIX
        prediction_array = np.matmul(X_train_array,B.transpose())
        #!print(prediction_array.shape)
        E_matrix_square = calculate_square_error_matrix(Y_train_array,prediction_array)
        E_matrix_absolute = calculate_absolute_error_matrix(Y_train_array,prediction_array)


        average_square_error_training = np.sum(E_matrix_square)/len(E_matrix_square)
        average_absolute_error_training = np.sum(E_matrix_absolute)/len(E_matrix_absolute)


        training_mse_scores += average_square_error_training
        training_mae_scores += average_absolute_error_training

        print(f"--------------------------------------------")
        print(f"FOLD NUMBER: {fold_index}")
        print(f"-----------------TRAINING-------------------")
        print(f"CURRENT AVERAGE SQUARE ERROR DURING TRAINING IN IS: {average_square_error_training}")
        print(f"CURRENT AVERAGE ABSOLUTE ERROR DURING TRAINING IN IS: {average_absolute_error_training}")


        Sum_of_squares = np.sum(E_matrix_square)
        #!print(Sum_of_squares)


        
        X_test_array = X_test.to_numpy()
        Y_test_array = y_test.to_numpy()

        prediction_array_test = np.matmul(X_test_array,B.transpose())

        E_matrix_square = calculate_square_error_matrix(Y_test_array,prediction_array_test)
        E_matrix_absolute = calculate_absolute_error_matrix(Y_test_array,prediction_array_test)
        average_square_error_testing = np.sum(E_matrix_square)/len(E_matrix_square)
        average_absolute_error_testing = np.sum(E_matrix_absolute)/len(E_matrix_absolute)
        
        print(f"-----------------TESTING--------------------")
        print(f"CURRENT AVERAGE SQUARE ERROR DURING TESTING IS: {average_square_error_testing}")
        print(f"CURRENT AVERAGE ABSOLUTE ERROR DURING TESTING IS: {average_absolute_error_testing}")
        
        testing_mse_scores += average_square_error_testing
        testing_mae_scores += average_absolute_error_testing



        fold_index += 1
    final_mse_training_score = training_mse_scores/num_folds
    final_mae_training_score = training_mae_scores/num_folds

    final_mse_testing_score = testing_mse_scores/num_folds
    final_mae_testing_score = testing_mae_scores/num_folds


    print(f"-----------TRAINING FINAL SCORES------------\nMSE: {final_mse_training_score},MAE: {final_mae_training_score}")
    print(f"-----------TESTING FINAL SCORES-------------\nMSE: {final_mse_testing_score},MAE: {final_mae_testing_score}")


    #*Here,after our k cross validation is finished,we train the model on the complete dataset one last time 
    x_final_array = X.to_numpy()
    y_final_array = y.to_numpy()
    
    final_B = calculate_slope_coefficient_matrix(x_final_array,y_final_array)
    print(f"--------------------------------------------")
    print(f"FINAL SLOPE COEFFICIENTS CALCULATED AT: {final_B}") 
    print(f"--------------------------------------------")

#!REMOVE ONLY FOR TESTING
df = pd.read_csv('pain.csv')
least_squares(df)