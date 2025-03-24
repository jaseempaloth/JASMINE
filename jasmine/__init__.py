"""
JASMINE - Python package for machine learning algorithms

This package provides tools and utilities for implementing various machine learning algorithms.
"""

__version__ = '0.1.0'
__author__ = 'Jaseem Paloth'


from .linear_model import LinearRegression, LogisticRegression
from .metrics import MSELoss, MAELoss, RMSELoss, CrossEntropyLoss

__all__ = ['LinearRegression',
            'LogisticRegression'
            'MSELoss',
            'MAELoss',
            'RMSELoss',
            'CrossEntropyLoss'
        ]

