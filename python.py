# Importing the necessary Libraries

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Loading the dataset and splitting the data into dependent and independent variables
data = pd.read_csv("D:\Regression\house_price_dataset.csv")
X = data.drop(columns=['house_price'])
Y = data['house_price']

# Checking missing values 
print(data.isnull().sum())

# Checking correlation
print(data.corr())

# Separating rows with missing data for the dataset
X_missing = X[X.isnull().any(axis=1)]
Y_missing = Y[X_missing.index]

# Non-missing dataset for training
X_full = X.dropna()
Y_full = Y[X_full.index]

# Splitting dataset into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X_full, Y_full, test_size=0.3, random_state=42)

# Initializing the Linear regression model
model = LinearRegression()

# Training the model
model.fit(X_train, Y_train)

# Making the prediction
Y_pred = model.predict(X_test)

# Evaluating the model
r2 = r2_score(Y_test, Y_pred)
mse = mean_absolute_error(Y_test, Y_pred)

print(f"R-squared error : {r2}")
print(f"Mean Squared Error : {mse}")

# Predicting the missing values from our dataset
X_missing_dataset = model.predict(X_missing)
print(f"Predictions for rows with missing data : {X_missing_dataset}")