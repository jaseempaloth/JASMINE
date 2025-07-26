# JASMINE - JAX Accelerated Statistical Models and Integrated Neural Engine

JASMINE is a lightweight machine learning library built on top of JAX, designed to leverage GPU and TPU acceleration for high-performance computing. The project aims to provide an accessible interface to JAX’s powerful capabilities while continuously updating with new features and models.

## Features

- Linear models (linear regression, logistic regression)
- Various loss functions (MSE, MAE, RMSE, Cross-Entropy)
- Automatic differentiation using JAX
- JIT compilation for improved performance

## Installation

```bash
# Clone the repository
git clone https://github.com/jaseempaloth/JASMINE

# Navigate to the directory
cd jasmine

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- JAX: Core library for automatic differentiation, accelerated numerical computing, and its NumPy-compatible API (`jnp`)

## Quick Start

### Linear Regression

```python
import jax.numpy as jnp
from jasmine import LinearRegression
from jasmine import MSELoss

# Create synthetic data
X = jnp.array([[1, 2], [3, 4], [5, 6]])
y = jnp.array([3, 7, 11])

# Create and fit the model
model = LinearRegression(loss=MSELoss)
model.fit(X, y, learning_rate=0.01, max_iter=1000)

# Make predictions
predictions = model.predict(X)
print(f"Predictions: {predictions}")

# Evaluate the model
r2_score = model.score(X, y)
print(f"R² Score: {r2_score}")
```

### Logistic Regression

```python
import jax.numpy as jnp
from jasmine import LogisticRegression
from jasmine import CrossEntropyLoss

# Create synthetic data
X = jnp.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = jnp.array([0, 0, 1, 1])

# Create and fit the model
model = LogisticRegression(loss=CrossEntropyLoss)
model.fit(X, y, learning_rate=0.1, max_iter=1000)

# Make predictions
predictions = model.predict(X)
print(f"Predictions: {predictions}")

# Get probabilities
probabilities = model.predict_proba(X)
print(f"Probabilities: {probabilities}")

# Evaluate the model
accuracy = model.score(X, y)
print(f"Accuracy: {accuracy}")
```

## License

[MIT License](LICENSE)
