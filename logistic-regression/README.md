# Logistic Regression Notes

## What is Logistic Regression?

Logistic Regression is a supervised machine learning algorithm used for classification problems. Unlike linear regression, which predicts continuous values, logistic regression predicts the probability that a given input belongs to a particular class (usually binary: 0 or 1, yes or no, true or false).

## What Problems Does It Solve?
- Binary classification (e.g., spam vs. not spam, disease vs. no disease)
- Multiclass classification (with extensions like softmax regression)
- Predicting the probability of an event occurring

---

## Types of Classification Problems and Logistic Regression

- **Binary Classification:**  
  Predicts one of two possible classes.  
  *Example:* Email spam detection (spam or not spam), disease prediction (disease or no disease).

- **Multiclass Classification:**  
  Predicts one of three or more possible classes.  
  *Example:* Handwritten digit recognition (digits 0–9), classifying types of flowers.

- **Multilabel Classification:**  
  Predicts multiple classes for each instance (less common for logistic regression, but possible with adaptations).  
  *Example:* Tagging a news article with multiple topics.

**How Logistic Regression Solves These:**
- For binary classification, logistic regression uses the sigmoid function to output a probability between 0 and 1, and applies a threshold (usually 0.5) to decide the class.
- For multiclass classification, logistic regression can be extended using the “one-vs-rest” (OvR) or “softmax” (multinomial) approach.
- For multilabel, multiple binary logistic regression models can be trained, one for each label.

**Examples:**
- Binary: Predicting if a customer will buy a product (yes/no).
- Multiclass: Classifying the species of an iris flower (setosa, versicolor, virginica).
- Multilabel: Assigning multiple genres to a movie (action, comedy, drama).


---

## What is the Sigmoid Function?

The **Sigmoid function** (also called the logistic function) is a mathematical function that maps any real-valued number into a value between 0 and 1. In logistic regression, it is used to convert the output of a linear equation (z) into a probability.

## Visualizing the Sigmoid Function

![Sigmoid Function Graph](https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg)

*The sigmoid function maps any real value to a value between 0 and 1, which is interpreted as probability in logistic regression.*

**Formula:**

p = 1 / (1 + exp(-z))

Where:
- p = predicted probability (between 0 and 1)
- z = linear combination of inputs and weights (z = w₀ + w₁x₁ + ... + wₙxₙ)

**Why is it important?**
- It allows logistic regression to output probabilities.
- It enables classification by applying a threshold (e.g., 0.5) to decide the class.

---

---

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
