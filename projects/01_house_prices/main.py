"""
Project 1: House Price Prediction
Goal: Understand ML workflow: Load → Explore → Train → Evaluate → Visualize
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("=" * 60)
print("HOUSE PRICE PREDICTION PROJECT")
print("=" * 60)

# STEP 1: Load data
print("\n[STEP 1] Loading California Housing Dataset...")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='Price')

print(f"Dataset shape: {X.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")

# STEP 2: Explore data
print("\n[STEP 2] Exploring Data...")
print("\nFirst 5 rows:")
print(X.head())

print("\nDataset statistics:")
print(X.describe())

# STEP 3: Split into train and test
print("\n[STEP 3] Splitting data into train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# STEP 4: Train model
print("\n[STEP 4] Training Linear Regression Model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("✓ Model trained!")

# STEP 5: Make predictions
print("\n[STEP 5] Making predictions on test set...")
y_pred = model.predict(X_test)
print("✓ Predictions complete!")

# STEP 6: Evaluate
print("\n[STEP 6] Evaluating Model...")
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"  MSE (Mean Squared Error):  {mse:.6f}")
print(f"  RMSE (Root Mean Squared Error): {rmse:.6f}")
print(f"  R² Score: {r2:.4f}")

# STEP 7: Visualize predictions
print("\n[STEP 7] Creating visualization...")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, s=20)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price', fontsize=12)
plt.ylabel('Predicted Price', fontsize=12)
plt.title('Actual vs Predicted House Prices', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('prediction_plot.png', dpi=100)
print("✓ Plot saved as 'prediction_plot.png'")

# STEP 8: Key findings
print("\n[STEP 8] Key Findings:")
print(f"1. Model explains {r2*100:.2f}% of variance in house prices")
print(f"2. Average prediction error: ${rmse*100_000:.2f}")
print(f"3. Model performs {'well' if r2 > 0.5 else 'poorly'} (R² > 0.5 is good)")
print(f"4. Data has {X.shape[1]} features affecting price")

print("\n" + "=" * 60)
print("PROJECT COMPLETE!")
print("=" * 60)
