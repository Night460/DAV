from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Sample data (Years of Experience, Number of Projects)
X = [
    [1, 2],
    [2, 3],
    [3, 5],
    [4, 7],
    [5, 8],
    [6, 10],
    [7, 12],
    [8, 14],
    [9, 15]
]

y = [35000, 40000, 50000, 55000, 60000, 70000, 75000, 80000, 85000]

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict salary
predicted = model.predict(X)

# Print coefficients
print("Intercept (b0):", model.intercept_)
print("Coefficients (b1, b2):", model.coef_)

# Plot actual vs predicted
plt.plot(y, label="Actual", marker='o')
plt.plot(predicted, label="Predicted", marker='x')
plt.title("Actual vs Predicted Salary")
plt.xlabel("Sample Index")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)
plt.show()
