# ==========================================
# AI-ML Assignment 3
# Salary Prediction using Polynomial Regression
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ----------------------------------------
# Task 1 : Data Understanding
# ----------------------------------------

df = pd.read_csv("Position_Salaries.csv")

print("First Five Records\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nInput Feature:")
print("Level")

print("\nTarget Variable:")
print("Salary")

# ----------------------------------------
# Task 2 : Data Preprocessing
# ----------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# Selecting the input feature and target variable
X = df[["Level"]]
y = df["Salary"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ----------------------------------------
# Task 3 : Model Development
# ----------------------------------------

poly = PolynomialFeatures(degree=3)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()

model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

# ----------------------------------------
# Task 4 : Model Evaluation
# ----------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========== Model Evaluation ==========\n")
print(f"Mean Absolute Error : {mae:.2f}")
print(f"Mean Squared Error  : {mse:.2f}")
print(f"R2 Score            : {r2:.4f}")

# Scatter Plot

plt.figure(figsize=(8,6))

plt.scatter(
    X,
    y,
    color="blue",
    label="Original Data"
)

# Smooth Curve

X_grid = np.arange(
    X.min().values[0],
    X.max().values[0],
    0.1
).reshape(-1,1)

plt.plot(
    X_grid,
    model.predict(poly.transform(X_grid)),
    color="red",
    linewidth=2,
    label="Polynomial Regression Curve"
)

plt.title("Salary Prediction using Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

plt.show()