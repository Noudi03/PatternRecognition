import random
import numpy as np

from .loss_functions import mean_square_error, mean_absolute_error
from .k_fold_CV import split_dataframe_into_folds


def drop_target_data(df):
    '''
        Args:
            df(pd.Dataframe):The dataset we gonna use as data for the model
        Returns:
            input_data(pd.Dataframe): A dataframe containing the data used as inputs for the model.
    '''
    # separating target from input data
    df = df.drop("median_house_value", axis=1)
    return (df)


def initialize_target_data(df):
    '''
        Args:
            df(pd.Dataframe):The dataset we gonna use as data for the model
        Returns:
            target_data(pd.Dataframe): A dataframe containing the data used by the error functions to train the model.
    '''
    value_data = df["median_house_value"].copy()

    # setting threshold as the median of all values
    threshold = value_data.median()

    # populating the target_data vector for use in the model with -1 being given to data
    # with median value below or equal the threshold an 1 for the opposite
    target_data = np.where(value_data > threshold, 1, -1)
    df.drop("median_house_value", axis=1)
    return (target_data)


def initialize_weight_data(weight_count):
    '''
        Args:
            weight_count(int): the number of weights the function shall initialize 

        Returns:
            weight_data(list): a list containing the initial weight data for the first iteration of the model.
    '''
    weight_data = []
    for weight in range(0, weight_count):
        weight_data.append(round(random.uniform(0, 1), 2))
    return weight_data


def calculate_sum(input_list, weight_list, bias):
    '''
        Args:
            input_list(list): a list containing the input data of our model.

            weight_list(list): a list containing all the weights of our model.

            bias(float): a number used by our machine learning algorithm to tweak how often our perceptron activates and/or stays inactive.
        Returns:
            sum(float)
    '''
    # *CALCULATING THE SUM
    # *MATHEMATICAL NOTATION OF FORMULA GOES AS FOLLOWS:
    # * X1*W1 + X2*W2 + X3*W3 + ... Xi*Wi + B
    # *WITH:
    # *X: input variable,comes from the training data list and there are as many of these as there are columns
    # *W: weight variable,comes from the weight data list and is initially initialized randomly with numbers between 0 and 1
    # *B: bias,is also randomly initialized and tweaked over time as the model trains

    sum = 0
    for i, input in enumerate(input_list):
        sum += input*weight_list[i]
    sum += bias
    return sum


def activation_function(weighted_sum, threshold):
    '''
        Args:
            weighted_sum(float): the sum calculated by a perceptron's weights and inputs plus the corresponding bias.
        Returns:
            prediction(int): the prediction of our model.Returns 1 if the weighted sum exceeds 0 and -1 if its lesser/equal to it.  
    '''
    if weighted_sum > threshold:
        return 1
    else:
        return -1


def update_weights(input_data, weight_data, prediction, target, learning_rate):
    '''
        Args:
            input_data(list): a list containing the input data for a certain sample of our data

            weight_data(list): a list containing the corresponding weights of a certain sample of our data

            prediction(float): the corresponding prediction our model made

            target(float): the correct answer we would like our model to give

            learning_rate(float): a value that determines how radically our model updates its weights in response to making errors.Making this value too big will make it overshoot often so its advised to keep it small(eg. 0.1,0.01 etc.).
        Returns:
            new_weights(list): a list containing the new updated weights of our model.
    '''
    new_weights = []

    # *FOR EACH WEIGHT WE SOLVE THE FOLLOWING FORMULA
    # * Wi' = Wi + a*(y-y')*Xi
    # * With:
    # * W: the i-th weight to be updated
    # * a: the learning rate of the algorithm
    # * y: the target value aka the right choice for the model to make
    # * y': the prediction the model actually made
    # * X: the i-th input that corresponds to the weight that is being updated.
    for i, weight in enumerate(weight_data):
        new_weights.append(weight + learning_rate *
                           (target-prediction)*input_data[i])
    return new_weights


def update_bias(bias, learning_rate, prediction, target):
    '''
        Args:
            bias(float):the bias to be updated.

            learning_rate(float): a value that determines how radically our model updates its weights in response to making errors.Making this value too big will make it overshoot often so its advised to keep it small(eg. 0.01,0.001 etc.).

            prediction(float): the corresponding prediction our model made

            target(float): the correct answer we would like our model to give
        Returns:
            new_bias(float): the new updated bias
    '''
    # *FOR EACH WEIGHT WE SOLVE THE FOLLOWING FORMULA
    # * B' = B + a*(y-y')
    # * With:
    # * B: the bias
    # * a: the learning rate of the algorithm
    # * y: the target value aka the right choice for the model to make
    # * y': the prediction the model actually made
    new_bias = bias + learning_rate*(target-prediction)
    return new_bias


def perceptron_algorithm(df, k=10, learning_rate=0.01):
    '''
        Args:
            df(pd.Dataframe): The dataframe we plan to train and test our model on.

            k(int): the number of folds to use for k-cross validation.Default set on 10

            learning_rate(float): a value that determines how radically our model updates its weights in response 
                                    to making errors.Making this value too big will make it overshoot often so its advised to keep it small(eg. 0.1,0.01 etc.).
                                    Default set to 0.01.
    '''

    # *Note: The dataset is already preprocessed and the missing values are filled with the median of the respective column.

    value_data = df["median_house_value"].copy()
    # setting threshold as the median of all values
    threshold = value_data.median()

    fold_list = split_dataframe_into_folds(df, k)

    target_data_lists = []
    for i, fold in enumerate(fold_list):
        target_data_lists.append(initialize_target_data(fold))

    input_data_lists = []
    for fold in fold_list:
        input_data_lists.append(drop_target_data(fold))

    square_loss_sum = 0
    absolute_loss_sum = 0
    for fold_index, fold in enumerate(input_data_lists):

        current_fold = []

        # initializing the training sets
        training_set = []
        training_target_set = []
        # initializing the testing sets
        testing_set = []
        testing_target_set = []

        # initializing bias and weights for each fold's model
        bias = round(random.uniform(-100, 100), 2)
        weight_data = initialize_weight_data(len(input_data_lists[0].columns))

        for index in range(0, len(fold_list)):
            if index != fold_index:
                current_fold = input_data_lists[index].values.tolist()
                current_target_data = target_data_lists[index]

                # adding the data to our training set as long as its not the current hold set
                # while also adding the equivalent target data to be used for the training and evaluation of the model
                training_set += current_fold

                # *NOTE: For some reason the target data seems to be a np array instead of a pd dataframe?Not certain why
                # *might end up moving this conversion to list to the initialize_target_data function and update the
                # *documentation there later
                training_target_set += list(current_target_data)
            else:
                current_fold = input_data_lists[index].values.tolist()
                current_target_data = target_data_lists[index]
                testing_set += current_fold
                testing_target_set += list(current_target_data)

        # iterating through the training set
        passed_training = False
        # calculating the predictions of the training set
        prediction = []
        for i, row in enumerate(training_set):
            # iterating through the elements of each row
            value_sum = calculate_sum(row, weight_data, bias)
            prediction.append(activation_function(value_sum, threshold))

        # calculating loss
        average_cost = mean_square_error(prediction, training_target_set)

        for i, row in enumerate(training_set):
            weight_data = update_weights(
                row, weight_data, prediction[i], training_target_set[i], learning_rate)
            bias = update_bias(bias, learning_rate,
                               prediction[i], training_target_set[i])

        prediction_tests = []

        for i, row in enumerate(testing_set):
            value_sum = calculate_sum(row, weight_data, bias)
            prediction_tests.append(activation_function(value_sum, threshold))
        # calculating loss
        average_cost_test = mean_square_error(
            prediction_tests, testing_target_set)
        average_absolute_cost = mean_absolute_error(
            prediction_tests, testing_target_set)
        square_loss_sum += average_cost_test
        absolute_loss_sum += average_absolute_cost
    square_loss_average = square_loss_sum/k
    absolute_loss_average = absolute_loss_sum/k
    print(f"--------------K-FOLD RESULTS----------------")
    print(f"{k}-FOLD VALIDATION AVERAGE MSE: {square_loss_average}")
    print(f"{k}-FOLD VALIDATION AVERAGE MAE: {absolute_loss_average}")
    print(f"--------------------------------------------\n")

    # *TRAINING ON ENTIRE DATASET AFTER K-CROSS VALIDATION HAS FINISHED RUNNING
    bias = round(random.uniform(-100, 100), 2)
    weight_data = initialize_weight_data(len(input_data_lists[0].columns))
    temp = df.drop('median_house_value', axis=1)
    input_data = temp.to_numpy()
    temp = df['median_house_value']
    target_data = temp.to_numpy()
    print(f"----------------FINAL MODEL-----------------")
    passed_training = False
    while passed_training == False:
        prediction = []

        # *MAKING PREDICTIONS
        for row in input_data:
            value_sum = calculate_sum(row, weight_data, bias)
            prediction.append(activation_function(value_sum, threshold))

        # *UPDATING WEIGHTS + BIAS
        for i, row in enumerate(input_data):
            weight_data = update_weights(
                row, weight_data, prediction[i], target_data[i], learning_rate)
            bias = update_bias(bias, learning_rate,
                               prediction[i], target_data[i])

        # *CALCULATING MEDIAN SQUARE ERROR(MSE) AND MEDIAN ABSOLUTE ERROR(MAE)
        mse = mean_square_error(prediction, target_data)
        mae = mean_absolute_error(prediction, target_data)
        if mse < 0.75:
            passed_training = True
            print(f"FINAL MSE: {mse},FINAL MAE: {mae}")
            print(f"--------------------------------------------\n")
