import pandas as pd
from sklearn.decomposition import PCA
import sys
import json

# Load data
csv_file_path = 'iot.csv'  # Update this path
try:
    data = pd.read_csv(csv_file_path, header=None, low_memory=False)  # Read the CSV file without header
except FileNotFoundError:
    sys.exit("CSV file not found.")
except Exception as e:
    sys.exit(f"An error occurred while reading the CSV file: {e}")

# Define column names based on the structure of the data
data.columns = ['ts', 'device', 'co', 'humidity', 'light', 'lpg', 'motion', 'smoke', 'temp']

# List of features to use for PCA analysis
features = ['co', 'humidity', 'light', 'lpg', 'motion', 'smoke', 'temp']

# Check if the required columns are present in the data
missing_cols = [col for col in features if col not in data.columns]
if missing_cols:
    sys.exit(f"Missing required columns in CSV file: {', '.join(missing_cols)}")

# Preprocess data: Attempt to convert non-numeric columns to numeric and handle errors
for col in features:
    if data[col].dtype == object:  # Check for object dtype which indicates string
        try:
            # Convert text-based categorical data to numeric using pandas' factorize method
            data[col], _ = pd.factorize(data[col])
        except ValueError as e:
            sys.exit(f"Error converting column '{col}' to numeric: {e}")

# Drop rows with NaN values in the selected features after preprocessing
data = data.dropna(subset=features)

# Check if data still contains valid numeric values
if data.empty or data[features].isna().any().any():
    sys.exit("Data cleaning error: Unable to retain valid numeric data.")

# Standardizing the features
x = data.loc[:, features].values

# PCA projection to 2D
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

# Combine with other columns if needed
finalDf = pd.concat([data[['ts']], principalDf], axis=1)

# Convert DataFrame to JSON
output = finalDf.to_json(orient='records')

# Print JSON output
print(output)


# Write JSON output to a file
output = finalDf.to_json(orient='records')

with open('output.json', 'w') as f:
    f.write(output)