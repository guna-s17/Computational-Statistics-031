import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Print the total number of rows and columns
print("Total number of rows and columns:")
print(iris_df.shape)

# Calculate mean and standard deviation for each column
mean_values = iris_df.mean()
sd_values = iris_df.std()

# Create a DataFrame with the statistics
stats_df = pd.DataFrame({
    'Mean': mean_values,
    'Standard Deviation': sd_values
})

# Print the statistics
print("\nMean and Standard Deviation for each attribute:")
print(stats_df)

# Plot only the normal distribution (KDE) for each feature
for column in iris_df.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(iris_df[column], kde=True, stat='density', linewidth=0, color='blue')
    plt.title(f'Normal Distribution of {column}')
    plt.tight_layout()
    plt.show()
