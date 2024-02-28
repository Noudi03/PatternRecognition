import math
import random
import pandas as pd
import numpy as np

from loss_functions import median_square_loss,median_absolute_loss



from k_fold_CV import split_dataframe_into_folds


def drop_target_data(df):
    '''
        INPUTS:
            df(pd.Dataframe):The dataset we gonna use as data for the model
        RETURNS:
            input_data(pd.Dataframe): A dataset containing the data used as inputs for the model.
    '''
    #separating target from input data
    df = df.drop("median_house_value", axis=1)


    return (df)



def initialize_target_data(df):
    '''
        INPUTS:
            df(pd.Dataframe):The dataset we gonna use as data for the model
        RETURNS:
            target_data(pd.Dataframe): A dataset containing the data used by the error functions to train the model.
    '''
    value_data = df["median_house_value"].copy()
    
    #setting threshold as the median of all values
    threshold = value_data.median()
    #print(threshold)
    #populating the target_data vector for use in the model with -1 being given to data 
    #with median value below or equal the threshold an 1 for the opposite 
    target_data = np.where(value_data > threshold, 1, -1)
    df.drop("median_house_value",axis=1);
    return(target_data)


def initialize_weight_data(weight_count):
    '''
        INPUTS:
            weight_count(int): the number of weights the function shall initialize 

        RETURNS:
            weight_data(list): A dataset containing the initial weight data for the first iteration of the model.
    '''
    weight_data = []
    for weight in range(0,weight_count):
        weight_data.append(round(random.uniform(0,1), 2))
    return weight_data


def calculate_sum(input_list,weight_list,bias):
    #*CALCULATING THE SUM
    #*MATHEMATICAL NOTATION OF FORMULA GOES AS FOLLOWS:
    #* X1*W1 + X2*W2 + X3*W3 + ... Xi*Wi + B
    #*WITH:
    #*X: input variable,comes from the training data list and there are as many of these as there are columns
    #*W: weight variable,comes from the weight data list and is initialisy initialized randomly with numbers between 0 and 1
    #*B: bias,is also randomly initialised and tweaked over time as the model trains

    sum = 0
    for i,input in enumerate(input_list):
        sum += input*weight_list[i]
    sum += bias
    return sum

def activation_function(weighted_sum):        
    if weighted_sum > 0:
        return 1
    else:
        return -1
    

    #TODO CHANGE FROM SIGMOID TO THE ONE HE WANTS AFTER YOU DONE TESTING
    '''
    if weighted_sum >= 0:
        z = math.exp(-weighted_sum)
        return 1 / (1 + z)
    else:
        z = math.exp(weighted_sum)
        return z / (1 + z)
    '''


def update_weights(input_data,weight_data,prediction,target,learning_rate):
    new_weights = []

    #*FOR EACH WEIGHT WE SOLVE THE FOLLOWING FORMULA
    #* Wi' = Wi + a*(y-y')*Xi
    #* With:
    #* W: the i-th weight to be updated
    #* a: the learning rate of the algorithm
    #* y: the target value aka the right choice for the model to make
    #* y': the prediction the model actually made
    #* X: the i-th input that corresponds to the weight that is being updated. 
    for i,weight in enumerate(weight_data):
        new_weights.append(weight + learning_rate*(target-prediction)*input_data[i])
    #print(f"NEW WEIGHTS: {new_weights}")
    return new_weights

def update_bias(bias,learning_rate,prediction,target):
    #*FOR EACH WEIGHT WE SOLVE THE FOLLOWING FORMULA
    #* B' = B + a*(y-y')
    #* With:
    #* B: the bias 
    #* a: the learning rate of the algorithm
    #* y: the target value aka the right choice for the model to make
    #* y': the prediction the model actually made
    new_bias = bias + learning_rate*(target-prediction)
    return new_bias

def perceptron(df,k=10,learning_rate = 0.1):
    '''
        INPUTS:
            input_data(pd.Dataframe): the input vector X for our model,containing values X1 through Xn.
            
            weight_data(pd.Dataframe): the weight vector W for our model,containing values W1 through Wn.                        
    '''

    #!USED TO OUTPUT INFO HERE 
    log = open("log.txt","w")
    log.write("Perceptron main function just begun execution\n")
    log.close()

    #*Note: The dataset is already preprocessed and the missing values are filled with the median of the respective column.    

    #TODO FIX OCEAN PROXIMITY AND REMOVE THIS LINE OF CODE AS SOON AS POSSIBLE
    #df = df.drop("ocean_proximity", axis=1)
    #log = open("log.txt","a")
    #log.write("DROPPED OCEAN PROXIMITY\n")
    #log.close()
    
    #First we randomise and split our dataset into k folds
    fold_list = split_dataframe_into_folds(df,k)
    log = open("log.txt","a")
    log.write(f"SPLIT DADAFRAME INTO {k} FOLDS\n")
    #then we initialize the matching target data for each fold
    #target_data_lists[0] should have the target data for fold_list[0] etc. etc.


    #TODO DONT FORGET TO UPDATE TARGET DATA LIST INITIALIZATION CODE SO THAT THE AVERAGE IS CALCULATED ON THE WHOLE DATASET NOT ON THE INDIVIDUAL ONE
    target_data_lists = []
    for i,fold in enumerate(fold_list):
        target_data_lists.append(initialize_target_data(fold))
        log.write(f"INITIALIZED TARGET DATA OF FOLD NO.{i}\n")    


    #!THIS IS HERE FOR TESTING ONLY DONT FORGET TO REMOVE LATER
    '''
    for fold_index,fold in enumerate(fold_list):
        current_fold = fold.values.tolist()
        print("fold with id: ",fold_index)
        for row_index,row in enumerate(current_fold): 
            if row_index < 10:
                print(row,"equivalent target data:",target_data_lists[fold_index][row_index])
    '''


    #then we get the input data by dropping the target data from the original dataset    
    input_data_lists = []
    for fold in fold_list:
        input_data_lists.append(drop_target_data(fold))
    log.write(f"PREPARED INPUT DATA LISTS\n")
    log.close()
    #iterating through each fold of the list
    #we use the input data since it is essentially the folded dataframe but with the target data dropped

    #!  COMMENTED THESE OUT IN CASE WE CHANGE OUR MINDS LATER
    #!bias = 0
    #!weight_data = []
    #!epoch_counter = 1
    #!epoch_loss_square = 0
    #!epoch_loss_absolute = 0

    square_loss_sum = 0
    absolute_loss_sum = 0
    for fold_index,fold in enumerate(input_data_lists):
        log = open("log.txt","a")
        log.write(f"CURRENT TESTING FOLD IS FOLD NO.{fold_index}\n")
        #! REMOVE THIS LATER ONLY USE FOR TESTING
        #print("Fold index is:",fold_index)

        current_fold = []
        

        #initilizing the training sets
        training_set = []
        training_target_set = []
        #initializing the testing sets
        testing_set = []
        testing_target_set = []

        #initializing bias and weights for each fold's model
        bias = round(random.uniform(-100,100), 2)
        weight_data = initialize_weight_data(len(input_data_lists[0].columns))  



        for index in range (0,len(fold_list)):
            if index != fold_index:
                current_fold = input_data_lists[index].values.tolist()
                current_target_data = target_data_lists[index]
                #! REMOVE THIS LATER ONLY USE FOR TESTING
                #print("Index is:",index)
                #print(current_fold)


                #adding the data to our training set as long as its not the current hold set
                #while also adding the equivalent target data to be used for the training and evaluation of the model
                training_set += current_fold 

                #*NOTE: For some reason the target data seems to be a np array instead of a pd dataframe?Not certain why
                #*might end up moving this conversion to list to the initialize_target_data function and update the 
                #*documentation there later
                training_target_set += list(current_target_data)                
            else:
                current_fold = input_data_lists[index].values.tolist()
                current_target_data = target_data_lists[index]
                testing_set += current_fold
                testing_target_set += list(current_target_data)
        
        #!COMMENTED OUT FOR NOW IN CASE WE DECIDE TO UNDO THIS CHANGE.DONT FORGET TO REMOVE LATER
        '''
        if bias == 0:
            bias = round(random.uniform(-100,100), 2)
        if len(weight_data) == 0:
            weight_data = initialize_weight_data(len(input_data_lists[0].columns))  
        '''
        #print(weight_data)
        log.write(f"ABOUT TO START THE TRAINING PROCESS WITH TESTING FOLD NO.{fold_index}\n")
        log.close()
        #iterating through the training set
        passed_training = False
        #calculating the predictions of the training set
        prediction = []
        for i,row in enumerate(training_set):
            #iterating through the elements of each row 
            value_sum = calculate_sum(row,weight_data,bias)
            #print(f"VALUE_SUM: {value_sum}")
            #!FOR TESTS ONLY REMOVE LATER
            #log = open("log.txt","a")
            #log.write(f"ROW_ID:{i},ROW:{row},WEIGHT:{weight_data},BIAS:{bias},SUM:{value_sum}")
            #log.close()
            prediction.append(activation_function(value_sum))
            #print(f"PREDICTION: {activation_function(value_sum)}")

        #calculating loss
        average_cost = median_square_loss(prediction,training_target_set)

        log = open("log.txt","a")

        #log.write(f"TEST FOLD INDEX: {fold_index},EPOCH:{epoch_counter}\n")
        
        for i,row in enumerate(training_set):
            #log.write(f"TEST ID:{i},ROW DATA: {row},WEIGHTS: {weight_data},PREDICTION:{prediction[i]},TARGET:{training_target_set[i]},LEARNING RATE:{learning_rate}\n")
            weight_data = update_weights(row,weight_data,prediction[i],training_target_set[i],learning_rate)
            bias = update_bias(bias,learning_rate,prediction[i],training_target_set[i])    
            log = open("log.txt","a")
            #log.write(f"EPOCH_COUNTER:{epoch_counter}\nWEIGHTS HAVE BEEN UPDATED TO:{weight_data}\nBIAS HAS BEEN UPDATED TO:{bias}\n")
            #print(f"WEIGHTS UPDATED TO: {weight_data},BIAS UPDATED TO: {bias}")
            
        #log.write(f"BEGINNING TESTING ON FOLD {fold_index}\n")

        prediction_tests = []

        #log.write(f"TESTING COMMENCING,WEIGHT DATA AT THE MOMENT IS: {weight_data}\n")
        #log.write(f"NUMBER OF ENTRIES IN TESTING SET: {len(testing_set)}\n")
        log.close()
        for i,row in enumerate(testing_set):
            #log = open("log.txt","a")
            #log.write(f"TEST ID:{i},ROW DATA: {row},PREDICTION:{prediction[i]},TARGET:{training_target_set[i]}")
            #log.close()
            value_sum = calculate_sum(row,weight_data,bias)
            prediction_tests.append(activation_function(value_sum))
        #calculating loss
        average_cost_test = median_square_loss(prediction_tests,testing_target_set)
        average_absolute_cost = median_absolute_loss(prediction_tests,testing_target_set)
        #!print(f"MEDIAN SQUARE LOSS: {average_cost_test}")
        square_loss_sum += average_cost_test
        absolute_loss_sum += average_absolute_cost


    #!epoch_counter += 1    
    #!epoch_loss_square += square_loss_sum/k
    #!epoch_loss_absolute += absolute_loss_sum/k
    #!log = open("log.txt","a")
    #!log.write(f"EPOCH {epoch_counter},SQUARE LOSS: {epoch_loss_square},ABSOLUTE LOSS: {epoch_loss_absolute}")
    '''
    if epoch_loss_square < 0.7:
        passed_training = True
    '''
    #TODO: ADD TEST AND TRAIN HERE

    #!FOR TESTING ONLY DO REMOVE LATER
    #print(f"TRAINING SET ENTRY COUNT: {len(training_set)}")
    #print(f"TESTING SET ENTRY COUNT: {len(testing_set)}")
    #print(f"TOTAL SET ENTRY COUNT: {len(training_set)+len(testing_set)}")


    #*TRAINING ON ENTIRE DATASET AFTER K-CROSS VALIDATION HAS FINISHED RUNNING
    bias = round(random.uniform(-100,100), 2)
    weight_data = initialize_weight_data(len(input_data_lists[0].columns))  
    temp = df.drop('median_house_value', axis=1)
    input_data = temp.to_numpy()
    temp = df['median_house_value']
    target_data = temp.to_numpy()

    passed_training = False
    while passed_training == False:
        prediction = []
        
        #*MAKING PREDICTIONS
        for row in input_data:
            value_sum = calculate_sum(row,weight_data,bias)
            prediction.append(activation_function(value_sum))

        #*UPDATING WEIGHTS + BIAS
        for i,row in enumerate(input_data):
            weight_data = update_weights(row,weight_data,prediction[i],target_data[i],learning_rate)
            bias = update_bias(bias,learning_rate,prediction[i],target_data[i])

        #*CALCULATING MEDIAN SQUARE ERROR(MSE) AND MEDIAN ABSOLUTE ERROR(MAE)
        mse = median_square_loss(prediction,target_data)
        mae = median_absolute_loss(prediction,target_data)
        print(f"FINAL MSE: {mse},FINAL MAE: {mae}")
        if mse < 0.8:
            passed_training = True





#! REMOVE THIS SHIT ITS ONLY FOR TESTING PURPOSES WRYYYYYYYYYYYYYYYYYYYYYYY
#loading the housing dataset with the filled median values
df = pd.read_csv('pain.csv')
perceptron(df,10)