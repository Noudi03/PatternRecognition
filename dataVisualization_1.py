import pandas as pd
import matplotlib.pyplot as plt

#loading the housing dataset with the filled mean values
df = pd.read_csv('housing_filled.csv')

#plotting the histogram for each one of the numerical columns
df.hist(bins=70, figsize=(20,15))
plt.show()

#calculating the frequency of each different possible value of the ocean proximity column
ocean_proximity = df['ocean_proximity'].value_counts()

#these 5 different values are the only possible ones, so a true histogram isn't the most visually appealing way to represent the data
ocean_proximity.plot(kind='barh')

#adding the title and labels
plt.ylabel('Ocean Proximity')
plt.xlabel('Frequency')
plt.title('Distribution of Ocean Proximity in the dataset')
plt.show()