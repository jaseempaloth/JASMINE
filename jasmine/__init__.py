"""
JASMINE - JAX Accelerated Statistical Models and Integrated Neural Engine

A lightweight, high-performance machine learning library built on JAX with GPU/TPU 
acceleration support for scalable ML workflows. JASMINE provides an accessible 
interface to JAX's powerful capabilities with automatic differentiation, JIT 
compilation, and hardware acceleration.

Features:
- Linear models (Linear Regression, Logistic Regression)
- Comprehensive loss functions (MSE, MAE, RMSE, Cross-Entropy)
- Data preprocessing utilities (train/test split)
- Automatic differentiation using JAX
- JIT compilation for improved performance
- GPU/TPU acceleration support
"""

__version__ = '0.1.0'
__author__ = 'Jaseem Paloth'
__email__ = 'jaseem@jaseempaloth.com'
__license__ = 'MIT'

# Core imports
from .metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from .selection import train_test_split
from .regression import LinearRegression
from .datasets import make_regression
# Public API
__all__ = [
    'LinearRegression',
    'train_test_split',
    'mean_squared_error',
    'mean_absolute_error',
    'root_mean_squared_error',
    'r2_score',
    'make_regression'
]

