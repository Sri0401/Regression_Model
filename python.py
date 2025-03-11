import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Determine the base directory of this script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define possible relative locations for the dataset
paths = [
    os.path.join(base_dir, "house_price_dataset.csv"),
    os.path.join(base_dir, "data", "house_price_dataset.csv")
]

# Select the first path where the file exists
dataset_path = next((path for path in paths if os.path.exists(path)), None)
if dataset_path is None:
    raise FileNotFoundError(f"Dataset not found. Searched in: {paths}")

# Load the dataset from the selected path
data = pd.read_csv(dataset_path)
print("Missing values per column:\n", data.isnull().sum())

# Ensure the target column exists
if 'house_price' not in data.columns:
    raise ValueError("Column 'house_price' not found in dataset.")

# Separate features and target
X = data.drop(columns='house_price')
y = data['house_price']

# Split data into rows with missing values and complete rows
X_missing = X[X.isnull().any(axis=1)]
X_complete = X.dropna()
y_complete = y.loc[X_complete.index]

# Confirm there is sufficient complete data to train the model
if len(X_complete) < 10:
    raise ValueError("Not enough complete data for training.")

# Train-test split on complete data
X_train, X_test, y_train, y_test = train_test_split(
    X_complete, y_complete, test_size=0.3, random_state=42
)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test data and evaluate the model
y_pred = model.predict(X_test)
print(f"R-squared Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")

# Predict values for rows with missing data (filling missing features with training means)
if not X_missing.empty:
    X_missing_filled = model.predict(X_missing.fillna(X_train.mean()))
    print("Predictions for rows with missing values:", X_missing_filled)
else:
    print("No missing values detected.")
