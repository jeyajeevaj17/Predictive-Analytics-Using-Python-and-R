# Experiment 1: Simple Linear Regression
# Predict salary from years of experience using a small illustrative dataset.
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)  # years of experience
y = np.array([30000, 35000, 40000, 45000, 50000, 55000], dtype=float)  # salary

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
new_experience = np.array([[7.0]])
new_salary = model.predict(new_experience)[0]

print("SIMPLE LINEAR REGRESSION")
print("Slope:", round(float(model.coef_[0]), 2))
print("Intercept:", round(float(model.intercept_), 2))
print("R-squared:", round(float(r2_score(y, predictions)), 3))
print("Predicted salary for 7 years of experience: Rs.", round(float(new_salary), 2))
