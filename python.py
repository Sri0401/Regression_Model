import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def evaluate_model():
    """
    Loads the dataset, trains a linear regression model, evaluates it, and prints output.
    Returns a dictionary with the R-squared and MAE values.
    """
    # Determine the directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define possible paths for the dataset
    paths = [
        os.path.join(base_dir, "house_price_dataset.csv"),
        os.path.join(base_dir, "data", "house_price_dataset.csv")
    ]
    
    # Find the first valid dataset path
    dataset_path = next((p for p in paths if os.path.exists(p)), None)
    if dataset_path is None:
        raise FileNotFoundError(f"Dataset not found. Searched in: {paths}")

    # Load the dataset
    data = pd.read_csv(dataset_path)
    print("Missing values per column:\n", data.isnull().sum())

    # Ensure the target column exists
    if 'house_price' not in data.columns:
        raise ValueError("Column 'house_price' not found in the dataset.")

    # Separate features (X) and target (y)
    X = data.drop(columns='house_price')
    y = data['house_price']

    # Separate rows with missing values for potential later prediction
    X_missing = X[X.isnull().any(axis=1)]
    X_complete = X.dropna()
    y_complete = y.loc[X_complete.index]

    if len(X_complete) < 10:
        raise ValueError("Not enough complete data for training.")

    # Split complete data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_complete, y_complete, test_size=0.3, random_state=42
    )

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R-squared Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.2f}")

    # Predict missing values by filling NA with training feature means
    if not X_missing.empty:
        predictions = model.predict(X_missing.fillna(X_train.mean()))
        print("Predictions for rows with missing values:", predictions)
    else:
        print("No missing values detected.")

    return {"r2": r2, "mae": mae}

# Run the evaluation if the script is executed directly.
if __name__ == "__main__":
    evaluate_model()
