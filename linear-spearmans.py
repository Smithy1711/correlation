import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Generate sample data
np.random.seed(42)  # For reproducibility
data_size = 100
x = np.random.uniform(0, 100, data_size)  # Random values between 0 and 100
y = 2 * x + np.random.normal(0, 10, data_size)  # A linear relationship with some noise

# Create a DataFrame to hold the data
data = pd.DataFrame({'X': x, 'Y': y})

# Calculate Spearman's correlation
spearman_corr, p_value = spearmanr(data['X'], data['Y'])

# Plot the data
plt.scatter(data['X'], data['Y'], alpha=0.7)
plt.title(f"Scatter Plot of X and Y\nSpearman's Correlation: {spearman_corr:.2f}")
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

spearman_corr, p_value
