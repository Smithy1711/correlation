import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Generate non-linear data
np.random.seed(42)  # For reproducibility
data_size = 100
x_non_linear = np.random.uniform(0, 100, data_size)
y_non_linear = np.sin(x_non_linear / 10) + np.random.normal(0, 0.1, data_size)  # Non-linear (sinusoidal) relationship with noise

# Create a DataFrame to hold the non-linear data
non_linear_data = pd.DataFrame({'X': x_non_linear, 'Y': y_non_linear})

# Calculate Spearman and Pearson correlations for the non-linear data
spearman_corr_non_linear, p_value_spearman = spearmanr(non_linear_data['X'], non_linear_data['Y'])
pearson_corr_non_linear = non_linear_data.corr(method='pearson').loc['X', 'Y']

# Plot the non-linear data
plt.scatter(non_linear_data['X'], non_linear_data['Y'], alpha=0.7)
plt.title(f"Scatter Plot of Non-linear X and Y\nSpearman: {spearman_corr_non_linear:.2f}, Pearson: {pearson_corr_non_linear:.2f}")
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

spearman_corr_non_linear, pearson_corr_non_linear
