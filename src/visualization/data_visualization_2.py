import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import construct_absolute_path

#loading the housing dataset with the filled median values
csv_file_path = construct_absolute_path('housing_filled.csv')
df = pd.read_csv(csv_file_path)

#setting the style for the plot
sns.set_theme(style='whitegrid')

def plot_variable_pairs(data, variables):
    """This function generates a scatter plot for a pair of 2 variables, or 
        a scatter matrix for pairs of multiple variables

    Args:
        data (pandas.DataFrame): Our dataset containing the variables' values
        variables (list): A list of the variables we want to plot
    
    Returns:
        None
    
    """
    
    #if we have a set of 2 variables, we will use the seaborn.scatterplot() function 
    #*NOTE:the alpha parameter controls the transparency of the points
    if len(variables) == 2:
        sns.scatterplot(data, x=variables[0], y=variables[1], alpha=0.3) 
        plt.title(f"Scatter Plot for {variables[0]} and {variables[1]}")
        
    #if we have more than 2 variables, we will create a scatter matrix using the seaborn.pairplot() function
    elif len(variables) > 2:
        pairs = sns.pairplot(data[variables], kind='scatter', height=2.5, aspect=1)
        pairs.figure.suptitle(f"Scatter Matrix Plots for {', '.join(variables)}", y=1)
    plt.show()


vars_to_plot_2 = ['total_rooms', 'total_bedrooms']
plot_variable_pairs(df, vars_to_plot_2)
vars_to_plot_3 = ['total_rooms', 'total_bedrooms', 'median_income']
plot_variable_pairs(df, vars_to_plot_3)
vars_to_plot_4 = ['total_rooms', 'total_bedrooms', 'median_income', 'median_house_value']
plot_variable_pairs(df, vars_to_plot_4)