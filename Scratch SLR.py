import matplotlib.pyplot as plt

# Sample data (YearsExperience vs Salary)
X = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [35000, 40000, 50000, 55000, 60000, 70000, 75000, 80000, 85000]

# Calculate the mean of X and y
mean_x = sum(X) / len(X)
mean_y = sum(y) / len(y)

# Calculate slope (b1) and intercept (b0)
numerator = sum((X[i] - mean_x) * (y[i] - mean_y) for i in range(len(X)))
denominator = sum((X[i] - mean_x) ** 2 for i in range(len(X)))
b1 = numerator / denominator
b0 = mean_y - b1 * mean_x

# Predict y using the equation y = b0 + b1*x
y_pred = [b0 + b1 * x for x in X]

# Plotting
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted Line')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression (from scratch)")
plt.legend()
plt.show()

# Print the equation
print(f"Intercept (b0): {b0}")
print(f"Slope (b1): {b1}")
print(f"Regression Equation: Salary = {b0:.2f} + {b1:.2f} * Experience")
