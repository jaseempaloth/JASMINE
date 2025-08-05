# JASMINE - JAX Accelerated Statistical Models and Integrated Neural Engine

JASMINE is a lightweight machine learning library built on top of JAX, designed to leverage GPU and TPU acceleration for high-performance computing. The project aims to provide an accessible interface to JAX’s powerful capabilities while continuously updating with new features and models.

## Features

- Modular and extensible framework supporting diverse machine learning algorithms
- Core components for model development, training, and evaluation
- Data preprocessing utilities
- Automatic differentiation powered by JAX
- JIT compilation for high-performance training and inference
- Seamless GPU/TPU acceleration for scalable computation
- Clean and intuitive API for educational, research, and custom ML use cases

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
import jax
import jax.numpy as jnp
from jasmine import LinearRegression, mean_squared_error
from jasmine.selection import train_test_split
from jasmine.datasets import generate_regression

# Generate synthetic regression data
X, y = generate_regression(
    n_samples=1000,
    n_features=20,
    n_informative=5,
    noise=0.1,
    bias=2.0,
    random_state=42
)

print(f"Generated data: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target range: [{jnp.min(y):.2f}, {jnp.max(y):.2f}]")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Testing set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

# Initialize and train the linear regression model
model = LinearRegression()
model.train(X_train, y_train)
print("Model trained successfully.")

# Compare learned vs true parameters
print("Learned parameters:", model.params["w"])
print("Learned Bias:", model.params["b"])

# Make predictions on the test set
predictions = model.inference(X_test)
print(f"Predictions shape: {predictions.shape}")


# Evaluate the model
mse = model.evaluate(X_test, y_test, metrics_fn=mean_squared_error)
r2 = model.evaluate(X_test, y_test)
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
```

### Logistic Regression

```python
import jax.numpy as jnp
from jasmine.classification import LogisticRegression
from jasmine.metrics import binary_cross_entropy, accuracy_score
from jasmine.selection import train_test_split
from jasmine.datasets import generate_classification

# Generate synthetic binary classification data
X, y = generate_classification(
    n_samples=1000,
    n_features=10,
    n_informative=3,
    n_redundant=2,
    n_classes=2,
    class_sep=1.0,
    shuffle=True,
    random_state=42
)
print(f"Generated data: {X.shape[0]} samples, {X.shape[1]} features ")
print(f"Target range: [{jnp.min(y):.2f}, {jnp.max(y):.2f}]")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Testing set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

# Initialize and train the logistic regression model
model = LogisticRegression()
model.train(X_train, y_train)
print("Model trained successfully.")

# Learned parameters
print("Learned parameters:", model.params["w"])
print("Learned Bias:", model.params.get("b", "No bias term"))

# Make predictions on the test set
logits = model.inference(X_test)
print(f"Logits shape: {logits.shape}")

# Evaluate the model
accuracy = model.evaluate(X_test, y_test, metrics_fn=accuracy_score)
print(f"Accuracy: {accuracy:.4f}")
```

## License

[MIT License](LICENSE)
