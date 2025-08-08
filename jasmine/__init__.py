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

__version__ = '0.1.0'
__author__ = 'Jaseem Paloth'
__email__ = 'jaseem@jaseempaloth.com'
__license__ = 'MIT'

# Core imports
from .regression import LinearRegression
from .classification import LogisticRegression, KNNClassifier
from .metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score, \
    binary_cross_entropy, categorical_cross_entropy, accuracy_score
from .selection import train_test_split
from .datasets import generate_regression, generate_polynomial, generate_classification
from .preprocessing import StandardScaler
# Public API
__all__ = [
    'LinearRegression',
    'LogisticRegression',
    'KNNClassifier',
    'train_test_split',
    'mean_squared_error',
    'mean_absolute_error',
    'root_mean_squared_error',
    'r2_score',
    'binary_cross_entropy',
    'categorical_cross_entropy',
    'accuracy_score',
    'generate_regression',
    'generate_polynomial',
    'generate_classification',
    'StandardScaler'
]

