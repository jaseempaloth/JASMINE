# Quick Start Guide

Get up and running with JASMINE in 5 minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/jaseempaloth/JASMINE.git
cd JASMINE

# Install dependencies
pip install -r requirements.txt

# Install JASMINE in development mode
pip install -e .
```

## Prerequisites

- Python 3.8 or higher
- JAX (automatically installed with requirements)
- NumPy >= 1.21.0

## Your First JASMINE Model

### 1. Import Required Modules

```python
import jax.numpy as jnp
from jasmine.regression import LinearRegression
from jasmine.classification import LogisticRegression
from jasmine.datasets import generate_regression, generate_classification
from jasmine.preprocessing import StandardScaler
from jasmine import train_test_split
```

### 2. Linear Regression Example

```python
# Generate sample data
X, y = generate_regression(n_samples=500, n_features=5, noise=0.1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression(learning_rate=0.1, n_epochs=1000)
model.train(X_train, y_train)

# Evaluate
r2_score = model.evaluate(X_test, y_test)
print(f"R² Score: {r2_score:.4f}")
```

### 3. Classification Example

```python
# Generate classification data
X, y = generate_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier
model = LogisticRegression(learning_rate=0.1, n_epochs=1000)
model.train(X_train_scaled, y_train)

# Get accuracy
from jasmine.metrics import accuracy_score
accuracy = model.evaluate(X_test_scaled, y_test, metrics_fn=accuracy_score)
print(f"Accuracy: {accuracy:.4f}")
```

## Key Concepts

### JAX Integration
JASMINE is built on JAX, which means:
- **Automatic GPU/TPU acceleration** when available
- **JIT compilation** for faster execution (after first run)
- **Automatic differentiation** for gradient computation

### Performance Tips
1. **First run is slower** due to JIT compilation
2. **Subsequent runs are much faster** (10-100x speedup)
3. **Use JAX arrays** (`jnp.array`) for best performance
4. **Scale your features** for optimal convergence

## What's Next?

- [Complete Examples](examples.md) - Detailed tutorials and use cases
- [API Reference](api.md) - Full documentation of all functions
- [Performance Guide](performance.md) - Benchmarks and optimization tips

## Need Help?

Check out the [FAQ](faq.md) or browse the [examples/](../examples/) directory for more code samples.
