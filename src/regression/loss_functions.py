def median_square_loss(prediction_data,target_data):
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
    #log = open("loss_calc.txt","w")
    #log.write(f"PREDICTION ENTRIES: {len(prediction_data)}\n")
    #log.write(f"PREDICTION DATA: {prediction_data}\n")
    loss = []
    #print(f"PREDICTION DATA:{prediction_data}")
    #print(f"ENTRY COUNT IN CURRENT PREDICTION DATA:{len(prediction_data)}")
    for index,prediction in enumerate(prediction_data):
        cost_sum += (target_data[index] - prediction)**2
        #print((target_data[index] - prediction)**2)
        loss.append((target_data[index] - prediction)**2)

    average = cost_sum/entry_num 
    #print(cost_sum,entry_num,average)
    #log.write(f"COST_SUM: {cost_sum},AVERAGE: {average}")
    #log.close()
    return (average,loss)

def median_absolute_loss(prediction_data,target_data):
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

    print(f"Median absolute loss average cost value: {cost_sum/entry_num}")
    average = cost_sum/entry_num 
    return (average,loss)



#TODO REMOVE THESE LATER THEY ARE ONLY FOR TESTING
'''
prediction = [1,0,0,1,1,1,1,1,1,0,0,0,0]
target =     [1,1,1,1,1,1,1,1,1,0,0,0,0]
median_absolute_loss(prediction,target)
median_square_loss(prediction,target)
'''