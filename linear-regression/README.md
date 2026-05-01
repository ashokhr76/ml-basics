# LinearRegression Executive Summary

**LinearRegression** is a machine learning algorithm used to model the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a straight line (in 2D) or a hyperplane (in higher dimensions) to the data. It is commonly used for predicting continuous values, such as prices, sales, or trends.

**When to use:**
- When you want to predict a numeric value based on one or more input features.
- When the relationship between variables is approximately linear.

**Simple Example:**
Suppose you want to predict house prices based on the size (in square feet):

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Example data
X = np.array([[1000], [1500], [2000], [2500]])  # House sizes
y = np.array([200000, 250000, 300000, 350000])  # Prices

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict price for a 1800 sq ft house
predicted_price = model.predict([[1800]])
print(predicted_price)
```
