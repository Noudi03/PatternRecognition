import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils import load_dataset


df = load_dataset('housing_final.csv')

# splitting the data into features and target
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']
print(df.columns)

# defining the MLP regressor
mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# initializing 10-fold cross-validation
num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

training_mse_scores = []
training_mae_scores = []
validation_mse_scores = []
validation_mae_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # training the model
    mlp.fit(X_train, y_train)

    # predictions on training set for training error
    train_predictions = mlp.predict(X_train)
    # predictions on test set for validation error
    test_predictions = mlp.predict(X_test)

    # calculating MSE and MAE for training and validation
    training_mse = mean_squared_error(y_train, train_predictions)
    training_mae = mean_absolute_error(y_train, train_predictions)
    validation_mse = mean_squared_error(y_test, test_predictions)
    validation_mae = mean_absolute_error(y_test, test_predictions)

    # appending scores to lists
    training_mse_scores.append(training_mse)
    training_mae_scores.append(training_mae)
    validation_mse_scores.append(validation_mse)
    validation_mae_scores.append(validation_mae)

# printing the average MSE and MAE for training and validation across all folds
print(f"Average Training MSE: {np.mean(training_mse_scores)}")
print(f"Average Training MAE: {np.mean(training_mae_scores)}")
print(f"Average Validation MSE: {np.mean(validation_mse_scores)}")
print(f"Average Validation MAE: {np.mean(validation_mae_scores)}")
