import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load preprocessed data
# Assuming 'x' is your preprocessed features data (e.g., after normalization or standardization)
csv_file_path = 'iot.csv'  # Update this path
data = pd.read_csv(csv_file_path)

# Define the features to use for PCA
features = ['co', 'humidity', 'light', 'lpg', 'motion', 'smoke', 'temp']

# Preprocess data: Drop rows with NaN values in the selected features after preprocessing
data = data.dropna(subset=features)

# Extract the features to be used in PCA
x = data.loc[:, features].values

# Perform PCA transformation
pca = PCA(n_components=2)  # Reduced to 2 components
principalComponents = pca.fit_transform(x)

# Create DataFrame for the PCA results
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

# Plot explained variance ratio
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.7, align='center')
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance Ratio of PCA Components")
plt.show()

# Plot 2D data points after PCA
plt.figure(figsize=(8, 6))
plt.scatter(principalComponents[:, 0], principalComponents[:, 1], alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D Plot of PCA Components")
plt.show()
