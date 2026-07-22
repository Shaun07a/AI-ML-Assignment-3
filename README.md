# Salary Prediction using Polynomial Regression

## Objective

The objective of this project is to develop a Polynomial Regression model that predicts employee salaries based on their position level. Since the relationship between position level and salary is non-linear, Polynomial Regression (Degree = 3) is used to improve prediction accuracy.

---

## Dataset Link

Position Salaries Dataset

https://www.kaggle.com/datasets/akram24/position-salaries

---

## Libraries Used

- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Methodology

1. Load the Position Salaries dataset using Pandas.
2. Display the first five records, dataset information, and summary statistics.
3. Check for missing values.
4. Select the input feature (`Level`) and target variable (`Salary`).
5. Split the dataset into 80% training data and 20% testing data.
6. Transform the input feature using Polynomial Features (Degree = 3).
7. Train a Polynomial Regression model.
8. Predict salaries for the testing dataset.
9. Evaluate the model using:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - R² Score
10. Visualize the original data and the Polynomial Regression curve.

---

## Results

The Polynomial Regression model successfully captures the non-linear relationship between position level and salary. The evaluation metrics indicate that the model predicts salaries accurately, and the regression curve closely follows the distribution of the original data.

---

## Conclusion

Polynomial Regression provides better prediction performance than Linear Regression for datasets with non-linear relationships. By transforming the input feature into polynomial terms, the model effectively captures the curved trend between position level and salary. This approach improves prediction accuracy while remaining relatively simple to implement and interpret.
