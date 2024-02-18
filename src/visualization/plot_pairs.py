import matplotlib.pyplot as plt
import seaborn as sns

def plot_variable_pairs(data, variables):
    """This function generates a scatter plot for a pair of 2 variables, or 
        a scatter matrix for pairs of multiple variables

    Args:
        data (pandas.DataFrame): Our dataset containing the variables' values
        variables (list): A list of the variables we want to plot
    
    Returns:
        None
    
    """
    #setting the style for the plot
    sns.set_theme(style='whitegrid')
    
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