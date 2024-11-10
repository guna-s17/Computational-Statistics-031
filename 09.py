import matplotlib.pyplot as plt

# Sample input data
product = ['computer', 'monitor', 'laptop', 'printer', 'tablet']
quantity = [320, 450, 300, 120, 280]

# Create a figure with two subplots
fig, axs = plt.subplots(2, figsize=(8, 6))

# Create the H-Plot on the first subplot
axs[0].barh(product, quantity)
axs[0].set_title('H-Plot of Product Quantities')
axs[0].set_xlabel('Quantity')
axs[0].set_ylabel('Product')

# Create the bar plot on the second subplot
axs[1].bar(product, quantity)
axs[1].set_title('Bar Plot of Product Quantities')
axs[1].set_xlabel('Product')
axs[1].set_ylabel('Quantity')
axs[1].tick_params(axis='x', rotation=45)  # rotate x-axis labels

# Layout so plots do not overlap
fig.tight_layout()

# Show the plot
plt.show()
