# JASMINE - JAX Accelerated Statistical Models and Integrated Neural Engine

JASMINE is a lightweight machine learning library built on top of JAX, designed to leverage GPU and TPU acceleration for high-performance computing. The project aims to provide an accessible interface to JAX’s powerful capabilities while continuously updating with new features and models.

## Features

- Linear models (linear regression, logistic regression)
- Various loss functions (MSE, MAE, RMSE, Cross-Entropy)
- Data preprocessing utilities (train/test split)
- Automatic differentiation using JAX
- JIT compilation for improved performance
- GPU/TPU acceleration support

## Installation

```bash
# Clone the repository
git clone https://github.com/jaseempaloth/JASMINE

# Navigate to the directory
cd JASMINE

# Install dependencies
pip install -r requirements.txt

# Install JASMINE package in development mode
pip install -e .
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

### Data Preprocessing

```python
import jax.numpy as jnp
from jasmine import train_test_split
import jax

# Generate synthetic classification data using mathematical functions
key = jax.random.PRNGKey(42)
n_samples = 150

# Create features using normal distribution
X = jax.random.normal(key, (n_samples, 4)) * 2 + 1

# Generate targets using mathematical relationships
# Class 0: when sum of first two features < 0
# Class 1: when 0 <= sum < 2
# Class 2: when sum >= 2
feature_sum = X[:, 0] + X[:, 1]
y = jnp.where(feature_sum < 0, 0, jnp.where(feature_sum < 2, 1, 2))

print(f"Generated data shape: {X.shape}")
print(f"Target distribution: {jnp.bincount(y)}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train target distribution: {jnp.bincount(y_train)}")
print(f"Test target distribution: {jnp.bincount(y_test)}")
```

## License

[MIT License](LICENSE)
