# Examples and Tutorials

This page contains detailed examples and tutorials for using JASMINE.

## Table of Contents

- [Linear Regression Examples](#linear-regression-examples)
- [Logistic Regression Examples](#logistic-regression-examples)
- [Data Preprocessing](#data-preprocessing)
- [Advanced Configuration](#advanced-configuration)
- [Model Comparison](#model-comparison)

---

## Linear Regression Examples

### Basic Linear Regression

```python
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

### Linear Regression with Regularization

```python
from jasmine import LinearRegression
from jasmine.datasets import generate_regression

# Generate data with noise
X, y = generate_regression(n_samples=500, n_features=50, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model with L2 regularization
model_l2 = LinearRegression(
    learning_rate=0.01,
    n_epochs=2000,
    l2_penalty=0.1,
    verbose=1
)

# Model with L1 regularization (Lasso)
model_l1 = LinearRegression(
    learning_rate=0.01,
    n_epochs=2000,
    l1_penalty=0.1,
    verbose=1
)

# Model with Elastic Net (L1 + L2)
model_elastic = LinearRegression(
    learning_rate=0.01,
    n_epochs=2000,
    l1_penalty=0.05,
    l2_penalty=0.05,
    verbose=1
)

# Train all models
models = {'L2': model_l2, 'L1': model_l1, 'Elastic': model_elastic}
for name, model in models.items():
    model.train(X_train, y_train)
    r2 = model.evaluate(X_test, y_test)
    print(f"{name} Regularization R²: {r2:.4f}")
```

### Advanced Optimization

```python
from jasmine import LinearRegression

# Adam optimizer with advanced settings
model = LinearRegression(
    learning_rate=0.001,
    n_epochs=2000,
    optimizer='adam',
    beta1=0.9,
    beta2=0.999,
    adaptive_lr=True,
    early_stopping_patience=100,
    normalize_features=True,
    verbose=1
)

# Train with validation data for early stopping
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

history = model.train(
    X_train, y_train,
    validation_data=(X_val, y_val)
)

# Plot training history
import matplotlib.pyplot as plt
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training History')
plt.show()
```

---

## Logistic Regression Examples

### Binary Classification

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
probabilities = model.predict_probabilities(X_test)
predictions = model.predict(X_test)

print(f"Logits shape: {logits.shape}")
print(f"Probabilities range: [{jnp.min(probabilities):.3f}, {jnp.max(probabilities):.3f}]")

# Evaluate the model
accuracy = model.evaluate(X_test, y_test, metrics_fn=accuracy_score)
print(f"Accuracy: {accuracy:.4f}")
```

### Classification with Regularization

```python
from jasmine.classification import LogisticRegression
from jasmine.metrics import accuracy_score

# Generate challenging classification data
X, y = generate_classification(
    n_samples=2000,
    n_features=50,
    n_informative=10,
    n_classes=2,
    class_sep=0.8,  # Harder separation
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model with L2 regularization
model = LogisticRegression(
    learning_rate=0.1,
    n_epochs=2000,
    l2_penalty=0.01,
    verbose=1
)

model.train(X_train, y_train)

# Detailed evaluation
accuracy = model.evaluate(X_test, y_test, metrics_fn=accuracy_score)
probabilities = model.predict_probabilities(X_test)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Average Confidence: {jnp.mean(jnp.maximum(probabilities, 1-probabilities)):.4f}")

# Confidence analysis
high_confidence = jnp.sum((probabilities > 0.8) | (probabilities < 0.2))
print(f"High Confidence Predictions: {high_confidence}/{len(probabilities)} ({100*high_confidence/len(probabilities):.1f}%)")
```

---

## Data Preprocessing

### Feature Scaling

```python
from jasmine.preprocessing import StandardScaler
from jasmine.datasets import generate_classification
import jax.numpy as jnp

# Generate data with different scales
X, y = generate_classification(n_samples=1000, n_features=5, random_state=42)

# Add features with different scales
X_scaled_up = X * jnp.array([1, 100, 10000, 1000000, 1])
print("Original feature ranges:")
for i in range(X_scaled_up.shape[1]):
    print(f"  Feature {i}: [{jnp.min(X_scaled_up[:, i]):.2e}, {jnp.max(X_scaled_up[:, i]):.2e}]")

# Apply StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_scaled_up)

print("\nAfter StandardScaler:")
print(f"  Mean: {jnp.mean(X_normalized, axis=0)}")
print(f"  Std:  {jnp.std(X_normalized, axis=0)}")

# Verify inverse transform
X_recovered = scaler.inverse_transform(X_normalized)
recovery_error = jnp.mean(jnp.abs(X_scaled_up - X_recovered))
print(f"\nRecovery error: {recovery_error:.2e}")
```

### Complete Preprocessing Pipeline

```python
from jasmine.preprocessing import StandardScaler
from jasmine.classification import LogisticRegression
from jasmine.datasets import generate_classification
from jasmine.metrics import accuracy_score

# Generate data
X, y = generate_classification(n_samples=2000, n_features=20, random_state=42)

# Create train/validation/test splits
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

print(f"Dataset splits:")
print(f"  Train: {X_train.shape[0]} samples")
print(f"  Validation: {X_val.shape[0]} samples") 
print(f"  Test: {X_test.shape[0]} samples")

# Preprocessing pipeline
scaler = StandardScaler()

# Fit on training data only
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train model on scaled data
model = LogisticRegression(learning_rate=0.1, n_epochs=1000, l2_penalty=0.01)
model.train(X_train_scaled, y_train)

# Evaluate on all sets
train_acc = model.evaluate(X_train_scaled, y_train, metrics_fn=accuracy_score)
val_acc = model.evaluate(X_val_scaled, y_val, metrics_fn=accuracy_score)
test_acc = model.evaluate(X_test_scaled, y_test, metrics_fn=accuracy_score)

print(f"\nModel Performance:")
print(f"  Train Accuracy: {train_acc:.4f}")
print(f"  Validation Accuracy: {val_acc:.4f}")
print(f"  Test Accuracy: {test_acc:.4f}")

# Check for overfitting
if train_acc - val_acc > 0.05:
    print("⚠️  Possible overfitting detected!")
else:
    print("✅ Model generalizes well")
```

---

## Advanced Configuration

### Custom Loss Functions

```python
from jasmine import LinearRegression

# Model with Huber loss (robust to outliers)
model_huber = LinearRegression(
    learning_rate=0.01,
    n_epochs=2000,
    loss_function='huber',
    huber_delta=1.0,  # Huber delta parameter
    verbose=1
)

# Model with MAE loss
model_mae = LinearRegression(
    learning_rate=0.01,
    n_epochs=2000,
    loss_function='mae',
    verbose=1
)

# Generate data with outliers
X, y = generate_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)

# Add some outliers
outlier_indices = jnp.array([10, 50, 100, 200, 300])
y = y.at[outlier_indices].add(10.0)  # Add large outliers

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Compare models
models = {
    'MSE': LinearRegression(loss_function='mse'),
    'MAE': model_mae,
    'Huber': model_huber
}

for name, model in models.items():
    model.train(X_train, y_train)
    r2 = model.evaluate(X_test, y_test)
    print(f"{name} Loss R²: {r2:.4f}")
```

### Hyperparameter Tuning

```python
from jasmine import LinearRegression
import itertools

# Define hyperparameter grid
learning_rates = [0.001, 0.01, 0.1]
l2_penalties = [0.0, 0.01, 0.1]
optimizers = ['sgd', 'momentum', 'adam']

# Generate data
X, y = generate_regression(n_samples=1000, n_features=15, noise=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

best_score = -float('inf')
best_params = None

print("Hyperparameter Tuning Results:")
print("-" * 50)

# Grid search
for lr, l2, opt in itertools.product(learning_rates, l2_penalties, optimizers):
    model = LinearRegression(
        learning_rate=lr,
        l2_penalty=l2,
        optimizer=opt,
        n_epochs=1000,
        verbose=0
    )
    
    model.train(X_train, y_train)
    score = model.evaluate(X_val, y_val)
    
    print(f"LR={lr:5.3f}, L2={l2:5.3f}, Opt={opt:8s} → R²={score:.4f}")
    
    if score > best_score:
        best_score = score
        best_params = (lr, l2, opt)

print(f"\nBest Parameters: LR={best_params[0]}, L2={best_params[1]}, Optimizer={best_params[2]}")
print(f"Best Score: {best_score:.4f}")
```

---

## Model Comparison

### JASMINE vs Sklearn

```python
from jasmine import LinearRegression as JasmineLinear, LogisticRegression as JasmineLogistic
from sklearn.linear_model import LinearRegression as SklearnLinear, LogisticRegression as SklearnLogistic
from sklearn.datasets import make_regression, make_classification
import time

def compare_linear_regression():
    """Compare JASMINE vs Sklearn Linear Regression."""
    print("Linear Regression Comparison")
    print("=" * 40)
    
    # Generate data
    X, y = make_regression(n_samples=5000, n_features=20, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # JASMINE Model
    start_time = time.time()
    jasmine_model = JasmineLinear(learning_rate=0.1, n_epochs=1000)
    jasmine_model.train(X_train, y_train)
    jasmine_r2 = jasmine_model.evaluate(X_test, y_test)
    jasmine_time = time.time() - start_time
    
    # Sklearn Model
    start_time = time.time()
    sklearn_model = SklearnLinear()
    sklearn_model.fit(X_train, y_train)
    sklearn_r2 = sklearn_model.score(X_test, y_test)
    sklearn_time = time.time() - start_time
    
    print(f"JASMINE - R²: {jasmine_r2:.4f}, Time: {jasmine_time:.3f}s")
    print(f"Sklearn - R²: {sklearn_r2:.4f}, Time: {sklearn_time:.3f}s")
    print(f"Performance difference: {jasmine_r2 - sklearn_r2:.4f}")
    
    return jasmine_r2, sklearn_r2

def compare_logistic_regression():
    """Compare JASMINE vs Sklearn Logistic Regression."""
    print("\nLogistic Regression Comparison")
    print("=" * 40)
    
    # Generate data
    X, y = make_classification(n_samples=5000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # JASMINE Model
    start_time = time.time()
    jasmine_model = JasmineLogistic(learning_rate=0.1, n_epochs=1000)
    jasmine_model.train(X_train, y_train)
    jasmine_acc = jasmine_model.evaluate(X_test, y_test, metrics_fn=accuracy_score)
    jasmine_time = time.time() - start_time
    
    # Sklearn Model
    start_time = time.time()
    sklearn_model = SklearnLogistic(max_iter=1000)
    sklearn_model.fit(X_train, y_train)
    sklearn_acc = sklearn_model.score(X_test, y_test)
    sklearn_time = time.time() - start_time
    
    print(f"JASMINE - Accuracy: {jasmine_acc:.4f}, Time: {jasmine_time:.3f}s")
    print(f"Sklearn - Accuracy: {sklearn_acc:.4f}, Time: {sklearn_time:.3f}s")
    print(f"Performance difference: {jasmine_acc - sklearn_acc:.4f}")
    
    return jasmine_acc, sklearn_acc

# Run comparisons
if __name__ == "__main__":
    compare_linear_regression()
    compare_logistic_regression()
```

---

## Tips and Best Practices

### Performance Optimization

1. **JIT Compilation**: First run is slower due to compilation, subsequent runs are much faster
2. **Data Types**: Use `jnp.float32` for memory efficiency on GPUs
3. **Batch Size**: Larger batches often perform better with JAX
4. **Feature Scaling**: Always scale features for optimal convergence

### Debugging

```python
# Enable verbose training
model = LinearRegression(verbose=1)

# Check gradient magnitudes
model.train(X_train, y_train)
print("Final gradient norms:", jnp.linalg.norm(model.params["w"]))

# Monitor training progress
history = model.train(X_train, y_train, validation_data=(X_val, y_val))
print("Training history:", history)
```

### Memory Management

```python
# For large datasets, consider batching
def batch_train(model, X, y, batch_size=1000):
    n_samples = X.shape[0]
    for i in range(0, n_samples, batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        model.train(batch_X, batch_y)
```

For more advanced examples, check the [examples/](../examples/) directory.
