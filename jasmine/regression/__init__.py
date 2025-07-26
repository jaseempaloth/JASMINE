"""
Regression module for JASMINE.

This module contains regression algorithms implemented using JAX for high-performance
machine learning with automatic differentiation and hardware acceleration.
"""

from .linear_regression import LinearRegression

__all__ = [
    'LinearRegression'
]