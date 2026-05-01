# Logistic Regression Notes

## What is Logistic Regression?

Logistic Regression is a supervised machine learning algorithm used for classification problems. Unlike linear regression, which predicts continuous values, logistic regression predicts the probability that a given input belongs to a particular class (usually binary: 0 or 1, yes or no, true or false).

## What Problems Does It Solve?
- Binary classification (e.g., spam vs. not spam, disease vs. no disease)
- Multiclass classification (with extensions like softmax regression)
- Predicting the probability of an event occurring

## Key Formulas
- **Sigmoid (Logistic) Function:**
  
  p = 1 / (1 + exp(-z))
  
  Where:
  - p = predicted probability (between 0 and 1)
  - z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ (linear combination of inputs and weights)

- **Decision Rule:**
  
  If p >= 0.5, predict class 1; else, predict class 0

- **Cost Function (Log Loss / Binary Cross-Entropy):**
  
  J = - (1/n) Σ [yᵢ log(pᵢ) + (1 - yᵢ) log(1 - pᵢ)]
  
  Where:
  - yᵢ = actual label (0 or 1)
  - pᵢ = predicted probability
  - n = number of samples

## Simple Example
Suppose you want to predict if a student passes (1) or fails (0) an exam based on hours studied:

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Example data
X = np.array([[1], [2], [3], [4], [5]])  # Hours studied
y = np.array([0, 0, 0, 1, 1])            # Pass (1) or Fail (0)

# Create and train the model
model = LogisticRegression()
model.fit(X, y)

# Predict probability of passing for 3 hours studied
prob = model.predict_proba([[3]])
print('Probability of passing:', prob[0][1])

# Predict class (pass/fail)
pred = model.predict([[3]])
print('Predicted class:', pred[0])
```

---

Logistic regression is widely used for classification tasks in many fields, including medicine, finance, and marketing.
