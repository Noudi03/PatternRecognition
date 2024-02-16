import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#loading the housing dataset with the filled mean values
df = pd.read_csv('housing_filled.csv')

#setting the style for the plot
sns.set_theme(style='whitegrid')

#creating the scatter plot using seaborn
#in this instance of the plot each dot corresponds to a single combination of 2 variables:
#on the x axis, the amount of total rooms, and on the y axis the amount of total bedrooms
sns.scatterplot(data=df, x='total_rooms', y='total_bedrooms', alpha=0.3)

#showing the plot
plt.show()

# #!TODO make this work for every pair 3 of variables, this is a test
# vars_to_plot = ['total_rooms', 'total_bedrooms', 'median_income']

# #plotting the scatter matrix between 3 variables
# sns.pairplot(df[vars_to_plot], kind='scatter')
# plt.show()

# #!TODO make this work for every pair of 4 variables
# vars_to_plot = ['total_rooms', 'total_bedrooms', 'median_income', 'median_house_value']

# #plotting the scatter matrix between 4 variables
# sns.pairplot(df[vars_to_plot])
# plt.show()

#!TODO: clean up code, write comments for the function 
def plot_variable_pairs(data, variables):
    print(len(variables))
    if len(variables) > 2:
        pairs = sns.pairplot(data[variables], kind='scatter')
        pairs.figure.suptitle(f"Scatter Matrix Plots for {', '.join(variables)}")

    else:
        sns.scatterplot(data, x=variables[0], y=variables[1], alpha=0.3)
        plt.title(f"Scatter Plot for {variables[0]} and {variables[1]}")
        
    plt.show()



vars_to_plot_2 = ['total_rooms', 'total_bedrooms']
plot_variable_pairs(df, vars_to_plot_2)
vars_to_plot_3 = ['total_rooms', 'total_bedrooms', 'median_income']
plot_variable_pairs(df, vars_to_plot_3)
vars_to_plot_4 = ['total_rooms', 'total_bedrooms', 'median_income', 'median_house_value']
plot_variable_pairs(df, vars_to_plot_4)