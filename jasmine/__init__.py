"""
JASMINE - JAX Accelerated Statistical Models and Integrated Neural Engine

A lightweight, high-performance machine learning library built on JAX with GPU/TPU
acceleration support for scalable ML workflows. JASMINE provides an accessible
interface to JAX's powerful capabilities with automatic differentiation, JIT
compilation, and hardware acceleration.

Features:
- Modular and extensible framework supporting diverse machine learning algorithms
- Core components for model development, training, and evaluation
- Data preprocessing utilities
- Automatic differentiation powered by JAX
- JIT compilation for high-performance training and inference
- Seamless GPU/TPU acceleration for scalable computation
- Clean and intuitive API for educational, research, and custom ML use cases
"""

__version__ = "0.1.0"
__author__ = "Jaseem Paloth"
__email__ = "jaseem@jaseempaloth.com"
__license__ = "MIT"

# Core imports
from .linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from .neighbors import KNNClassifier
from .svm import SVMClassifier
from .losses import (
    bce_loss,
    categorical_ce_loss,
    hinge_loss,
    huber_loss,
    mae_loss,
    mse_loss,
    squared_hinge_loss,
)
from .metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    binary_cross_entropy,
    categorical_cross_entropy,
    accuracy_score,
)
from .model_selection import train_test_split
from .optim import Adam, Momentum, SGD
from .datasets import generate_regression, generate_polynomial, generate_classification
from .preprocessing import StandardScaler, OneHotEncoder

# Public API
__all__ = [
    "Lasso",
    "LinearRegression",
    "LogisticRegression",
    "Ridge",
    "ElasticNet",
    "KNNClassifier",
    "SVMClassifier",
    "SGD",
    "Momentum",
    "Adam",
    "train_test_split",
    "mse_loss",
    "mae_loss",
    "huber_loss",
    "bce_loss",
    "categorical_ce_loss",
    "hinge_loss",
    "squared_hinge_loss",
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error",
    "r2_score",
    "binary_cross_entropy",
    "categorical_cross_entropy",
    "accuracy_score",
    "generate_regression",
    "generate_polynomial",
    "generate_classification",
    "StandardScaler",
    "OneHotEncoder",
]
