import pandas as pd
import numpy as np

#!ΠΡΕΠΕΙ ΝΑ ΑΝΑΦΕΡΩ ΤΗΝ ΑΚΡΙΒΕΙΑ ΠΑΛΙΝΔΡΟΜΗΣΗΣ ΣΕ ΟΡΟΥΣ
#!ΜΕΣΟΥ ΤΕΤΡΑΓΩΝΙΚΟΥ ΣΦΑΛΜΑΤΟΣ ΚΑΙ ΜΕΣΟΥ ΑΠΟΛΥΤΟΥ ΣΦΑΛΜΑΤΟΣ
#!ΚΑΙ ΣΤΗΝ ΦΑΣΗ ΤΗΣ ΕΚΠΑΙΔΕΥΣΗΣ ΚΑΙ ΣΤΗΝ ΦΑΣΗ ΤΟΥ ΕΛΕΓΧΟΥ
#!ΣΥΜΦΩΝΑ ΜΕ ΤΗΝ ΜΕΘΟΔΟ 10 FOLD CROSS VALIDATION

#!TODO SPLIT THE DATASET INTO TRAINING AND TESTING SETS
#NOTE: THE IMPLEMENTATION OF THE LOOP IS NOT CORRECT, ONLY THERE TO CHECK IF THE GENERAL IDEA WORKS

#loading the housing dataset with the filled median values
df = pd.read_csv('../data/housing_filled.csv')

#*Note: The dataset is already preprocessed and the missing values are filled with the median of the respective column.
#a linear regression function g(x)= wx + b
#where: w is the weight vector
#       x is the feature vector
#       b is the bias term

#selecting the target variable and converting it to binary labels 
#(1 for above the median, -1 for below the median)
target_variable = df["median_house_value"].copy()
threshold = target_variable.median()
target_variable = np.where(target_variable >= threshold, 1, -1)

#dropping the target variable from the DataFrame
df = df.drop("median_house_value", axis=1)

#dropping the 'ocean_proximity' variable, since it is categorical
df = df.drop("ocean_proximity", axis=1)

#standardizing the input variables(z-score normalization)
#subtracting the mean and dividing by the standard deviation of each column
df = (df - df.mean()) / df.std()

#add a bias term to the input variables
df["bias"] = 1

# Convert the DataFrame to a Numpy array
input_features = df.values

# Initialize the weights
weights = np.zeros(input_features.shape[1])

# Set the learning rate
learning_rate = 0.01

# Keep track of the number of iterations
iteration = 0

# Repeat until convergence
converged = False
while not converged:
    converged = True
    correct_predictions = 0
    for i in range(target_variable.shape[0]):
        # Predict the class label for the current example
        target_prediction = np.where(np.dot(input_features[i, :], weights) >= 0, 1, -1)

        # Update the weights if the prediction is incorrect
        if target_variable[i] != target_prediction:
            converged = False
            weights += learning_rate * target_variable[i] * input_features[i, :]
        else:
            correct_predictions += 1
    iteration += 1
    accuracy = correct_predictions / target_variable.shape[0]
    print(f"Iteration {iteration}: weights: {weights} accuracy {round(accuracy, 4)} %")
    if converged:
        break

print(f"The Perceptron Algorithm has converged after {iteration} iterations.")
print(f"The weights of the model are: {weights}")