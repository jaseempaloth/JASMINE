# API Reference

Simple and comprehensive API documentation for JASMINE.

## Table of Contents

- [Linear Models](#linear-models)
- [Classification](#classification)
- [Preprocessing](#preprocessing)
- [Datasets](#datasets)
- [Metrics](#metrics)
- [Selection](#selection)

---

## Linear Models

### LinearRegression

```python
from jasmine.regression import LinearRegression
```

#### Constructor
```python
LinearRegression(
    use_bias=True,
    learning_rate=0.01,
    n_epochs=1000,
    loss_function=mean_squared_error,
    l1_penalty=0.0,
    l2_penalty=0.0
)
```

**Parameters:**
- `use_bias` (bool): Whether to include bias term. Default: True
- `learning_rate` (float): Learning rate for optimization. Default: 0.01
- `n_epochs` (int): Number of training epochs. Default: 1000
- `loss_function` (callable): Loss function to use. Default: mean_squared_error
- `l1_penalty` (float): L1 regularization strength. Default: 0.0
- `l2_penalty` (float): L2 regularization strength. Default: 0.0

#### Methods

**train(X, y, validation_data=None, early_stopping_patience=None, verbose=1)**
```python
model.train(X, y)
```
- `X`: Feature matrix (n_samples, n_features)
- `y`: Target values (n_samples,)
- `validation_data`: Optional (X_val, y_val) tuple for early stopping
- `early_stopping_patience`: Number of epochs to wait before stopping
- `verbose`: Verbosity level (0=silent, 1=progress)

**inference(X)**
```python
predictions = model.inference(X)
```
- Returns: Predicted values

**evaluate(X, y, metrics_fn=r2_score)**
```python
score = model.evaluate(X, y)  # Returns R²
mse = model.evaluate(X, y, metrics_fn=mean_squared_error)
```
- Returns: R² score by default, or custom metric

#### Attributes
- `params`: Model parameters (weights and bias)
- `use_bias`: Whether bias is used
- `learning_rate`: Current learning rate

---

## Classification

### LogisticRegression

```python
from jasmine.classification import LogisticRegression
```

#### Constructor
```python
LogisticRegression(
    use_bias=True,
    learning_rate=0.01,
    n_epochs=1000,
    loss_function=binary_cross_entropy,
    l1_penalty=0.0,
    l2_penalty=0.0
)
```

**Parameters:**
- `use_bias` (bool): Whether to include bias term. Default: True
- `learning_rate` (float): Learning rate for optimization. Default: 0.01
- `n_epochs` (int): Number of training epochs. Default: 1000
- `loss_function` (callable): Loss function to use. Default: binary_cross_entropy
- `l1_penalty` (float): L1 regularization strength. Default: 0.0
- `l2_penalty` (float): L2 regularization strength. Default: 0.0

#### Methods

**train(X, y, validation_data=None, early_stopping_patience=None, verbose=1)**
```python
model.train(X, y)
```
- `X`: Feature matrix (n_samples, n_features)
- `y`: Binary target values (n_samples,)
- `validation_data`: Optional (X_val, y_val) tuple for early stopping
- `early_stopping_patience`: Number of epochs to wait before stopping
- `verbose`: Verbosity level (0=silent, 1=progress)

**inference(X, threshold=0.5)**
```python
predictions = model.inference(X)
```
- Returns: Binary predictions (0 or 1)
- `threshold`: Decision threshold for classification

**predict_probabilities(X)**
```python
probabilities = model.predict_probabilities(X)
```
- Returns: Predicted probabilities

**evaluate(X, y, metrics_fn=accuracy_score)**
```python
accuracy = model.evaluate(X, y)
```
- Returns: Accuracy by default, or custom metric

---

## Preprocessing

### StandardScaler

```python
from jasmine.preprocessing import StandardScaler
```

#### Constructor
```python
StandardScaler(epsilon=1e-8)
```

**Parameters:**
- `epsilon` (float): Small value to avoid division by zero. Default: 1e-8

#### Methods

**fit(X)**
```python
scaler.fit(X)
```
- Compute mean and standard deviation from training data
- Returns: self (fitted scaler instance)

**transform(X)**
```python
X_scaled = scaler.transform(X)
```
- Apply scaling to data
- Requires scaler to be fitted first

**fit_transform(X)**
```python
X_scaled = scaler.fit_transform(X)
```
- Fit and transform in one step

**inverse_transform(X)**
```python
X_original = scaler.inverse_transform(X_scaled)
```
- Reverse the scaling

#### Attributes
- `params`: Scaling parameters (mean and scale)
- `epsilon`: Numerical stability parameter
- `is_fitted`: Property indicating if scaler is fitted

---

## Datasets

### generate_regression

```python
from jasmine.datasets import generate_regression

X, y = generate_regression(
    n_samples=100,
    n_features=20,
    n_informative=10,
    noise=0.0,
    bias=0.0,
    shuffle=True,
    coef=False,
    random_state=None
)
```

**Parameters:**
- `n_samples` (int): Number of samples. Default: 100
- `n_features` (int): Total number of features. Default: 20
- `n_informative` (int): Number of informative features. Default: 10
- `noise` (float): Standard deviation of Gaussian noise. Default: 0.0
- `bias` (float): Bias term in the underlying linear model. Default: 0.0
- `shuffle` (bool): Whether to shuffle features. Default: True
- `coef` (bool): If True, return ground truth coefficients. Default: False
- `random_state` (int): Random seed for reproducibility. Default: None

**Returns:**
- `X`: Feature matrix (n_samples, n_features)
- `y`: Target values (n_samples,)
- `coef`: Ground truth coefficients (if coef=True)

### generate_classification

```python
from jasmine.datasets import generate_classification

X, y = generate_classification(
    n_samples=100,
    n_features=20,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    class_sep=1.0,
    feature_noise=1.0,
    redundant_noise=0.0,
    shuffle=True,
    random_state=None
)
```

**Parameters:**
- `n_samples` (int): Number of samples. Default: 100
- `n_features` (int): Total number of features. Default: 20
- `n_informative` (int): Number of informative features. Default: 5
- `n_redundant` (int): Number of redundant features. Default: 2
- `n_classes` (int): Number of classes. Default: 2
- `class_sep` (float): Class separation factor. Default: 1.0
- `feature_noise` (float): Standard deviation of feature noise. Default: 1.0
- `redundant_noise` (float): Standard deviation of redundant features noise. Default: 0.0
- `shuffle` (bool): Whether to shuffle features. Default: True
- `random_state` (int): Random seed for reproducibility. Default: None

**Returns:**
- `X`: Feature matrix (n_samples, n_features)
- `y`: Target labels (n_samples,)

### generate_polynomial

```python
from jasmine.datasets import generate_polynomial

X, y = generate_polynomial(...)
```

*Note: Check the actual implementation for complete parameter list*

---

## Metrics

### Regression Metrics

```python
from jasmine.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
```

**mean_squared_error(y_true, y_pred)**
- Returns: Mean squared error

**mean_absolute_error(y_true, y_pred)**
- Returns: Mean absolute error

**root_mean_squared_error(y_true, y_pred)**
- Returns: Root mean squared error

**r2_score(y_true, y_pred)**
- Returns: R² coefficient of determination

### Classification Metrics

```python
from jasmine.metrics import accuracy_score, binary_cross_entropy, categorical_cross_entropy
```

**accuracy_score(y_true, y_pred)**
- Returns: Classification accuracy

**binary_cross_entropy(y_true, y_pred, from_logits=False)**
- `from_logits`: If True, y_pred contains logits; if False, probabilities
- Returns: Binary cross-entropy loss

**categorical_cross_entropy(y_true, y_pred, from_logits=False)**
- `from_logits`: If True, y_pred contains logits; if False, probabilities  
- Returns: Categorical cross-entropy loss

---

## Selection

### train_test_split

```python
from jasmine.selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=True,
    random_state=None
)
```

**Parameters:**
- `X`: Feature matrix
- `y`: Target values
- `test_size` (float): Proportion for test set (0.0 to 1.0). Default: 0.2
- `shuffle` (bool): Whether to shuffle data before splitting. Default: True
- `random_state` (int): Random seed for reproducibility. Default: None

**Returns:**
- `X_train, X_test, y_train, y_test`: Split datasets

---

## Quick Examples

### Linear Regression
```python
from jasmine.regression import LinearRegression
from jasmine.datasets import generate_regression

# Generate data
X, y = generate_regression(n_samples=100, n_features=5)

# Train model
model = LinearRegression()
model.train(X, y)

# Make predictions
predictions = model.inference(X)
```

### Logistic Regression
```python
from jasmine.classification import LogisticRegression
from jasmine.datasets import generate_classification

# Generate data
X, y = generate_classification(n_samples=100, n_features=5)

# Train model
model = LogisticRegression()
model.train(X, y)

# Make predictions
predictions = model.inference(X)
probabilities = model.predict_probabilities(X)
```

### Logistic Regression
```python
from jasmine.classification import LogisticRegression
from jasmine.datasets import generate_classification

# Generate data
X, y = generate_classification(n_samples=100, n_features=5)

# Train model
model = LogisticRegression()
model.train(X, y)

# Make predictions
predictions = model.predict(X)
probabilities = model.predict_probabilities(X)
```

### Preprocessing
```python
from jasmine.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Transform new data
X_new_scaled = scaler.transform(X_new)
```

---

## Common Patterns

### Complete Pipeline
```python
from jasmine.regression import LinearRegression
from jasmine.preprocessing import StandardScaler
from jasmine.selection import train_test_split
from jasmine.datasets import generate_regression

# 1. Generate/load data
X, y = generate_regression(n_samples=1000, n_features=10)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train model
model = LinearRegression(learning_rate=0.01, n_epochs=1000)
model.train(X_train_scaled, y_train)

# 5. Evaluate
score = model.evaluate(X_test_scaled, y_test)
print(f"R² Score: {score:.4f}")
```

### Model Comparison
```python
from jasmine.regression import LinearRegression

models = {
    'Linear': LinearRegression(),
    'Ridge': LinearRegression(l2_penalty=0.1),
    'Lasso': LinearRegression(l1_penalty=0.1)
}

for name, model in models.items():
    model.train(X_train, y_train)
    score = model.evaluate(X_test, y_test)
    print(f"{name}: {score:.4f}")
```

For detailed examples, see [examples.md](examples.md).
