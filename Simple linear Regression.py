from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data (YearsExperience vs Salary)
X = [[1], [2], [3], [4], [5], [6], [7], [8], [9]]
y = [35000, 40000, 50000, 55000, 60000, 70000, 75000, 80000, 85000]

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict
predicted = model.predict(X)

# Plot
plt.scatter(X, y, color='blue')          # Actual data
plt.plot(X, predicted, color='red')      # Regression line
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression")
plt.show()

# Print coefficients
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])
