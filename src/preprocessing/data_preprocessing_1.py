from ..utils.utils import load_dataset, get_numerical_features, get_categorical_features

#loading the housing dataset
df = load_dataset('housing.csv')

#calling the functions to get the numerical and categorical features
numerical_features = get_numerical_features(df)
categorical_features = get_categorical_features(df)

#displaying the results
print("The numerical features are : ", numerical_features)
print("The categorical_features are:", categorical_features)