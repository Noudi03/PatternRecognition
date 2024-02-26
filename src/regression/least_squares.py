import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from numpy.linalg import inv

def least_squares(input_data):
    '''
        INPUTS:
            input_data(): 
        RESULTS:

    '''
    #!REMOVE ONLY FOR TESTING
    num_folds = 10
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)


    #splitting the data into features and target
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']


    #*initializing the score variables
    training_mse_scores = []
    training_mae_scores = []
    validation_mse_scores = []
    validation_mae_scores = []

    #*train_index refers to the indeces of our current train folds,while the test_index to the index of our current test fold
    #*We gonna initialize 2 different sets of numpy arrays here.One being X,representing the input and the other being Y,representing the target values.
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
        #* THE FORMULA TO CALCULATE THE B MATRIX IS:
        #* B = (inverse((transpose(X)*X))) * transpose(X)*Y

        X_transpose = X_train_array.transpose()
        #* temp_A = (transpose(X)*X)^-1)
        temp_A = inv(np.matmul(X_transpose,X_train_array)) 
        #* temp_B = transpose(X)*Y
        temp_B = np.matmul(X_transpose,Y_train_array)
        B = np.matmul(temp_A,temp_B)
        #!print(B.shape)
        #*CALCULATING THE PREDICTIONS:
        #*REMINDER THAT FOR PREDICTION Y' AND LINE THAT STARTS AT (0,0) OUR FORMULA IS Y' = TRANSPOSE(X)*B with B BEING THE SLOPE COEFFICIENTS OF OUR PARAMETERS IN THE X MATRIX
        prediction_array = np.matmul(X_train_array,B.transpose())
        print(prediction_array.shape)

        #*FOR COST CALCULATIONS TO INITIALIZE THE E MATRIX WE WILL USE THE SQUARE LOSS ALGORITHM,WITH FORMULA BEING:
        #*E = (Y - Y')^2 
        #!E = (Y_train_array - prediction_array)




#!REMOVE ONLY FOR TESTING
df = pd.read_csv('pain.csv')
least_squares(df)