import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Define possible file paths and check if the dataset exists
file_paths = ["house_price_dataset.csv", "data/house_price_dataset.csv", "./house_price_dataset.csv"]

file_path = None
for path in file_paths:
    if os.path.exists(path):
        file_path = path
        break

if file_path is None:
    raise FileNotFoundError(f"Dataset not found. Searched in: {file_paths}")

# Loading the dataset
data = pd.read_csv(file_path)

# Checking for missing values
print("Missing values per column:\n", data.isnull().sum())

# Separating independent and dependent variables
X = data.drop(columns=['house_price'])
Y = data['house_price']

# Separating missing and non-missing data
X_missing = X[X.isnull().any(axis=1)]  # Rows with missing values
Y_missing = Y.loc[X_missing.index]  # Corresponding target values

# Non-missing dataset for training
X_full = X.dropna()
Y_full = Y.loc[X_full.index]

# Ensure there's enough data to split
if len(X_full) < 10:
    raise ValueError("Not enough complete data for training. Consider handling missing values differently.")

# Splitting dataset into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_full, test_size=0.3, random_state=42)

# Initializing and training the Linear Regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Making predictions
Y_pred = model.predict(X_test)

# Evaluating the model
r2 = r2_score(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)

print(f"R-squared Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Predicting missing values if any
if not X_missing.empty:
    X_missing_filled = model.predict(X_missing.fillna(X_train.mean()))  # Fill NA with mean before predicting
    print(f"Predictions for rows with missing data:\n{X_missing_filled}")
else:
    print("No missing values to predict.")

