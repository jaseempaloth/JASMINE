Dataset Generators
==================

The :mod:`jasmine.datasets` module provides synthetic data generators for testing and examples.

.. currentmodule:: jasmine.datasets

Functions
---------

.. autosummary::
   :toctree: generated/

   generate_regression
   generate_classification
   generate_polynomial

generate_regression
-------------------

.. autofunction:: generate_regression

generate_classification
-----------------------

.. autofunction:: generate_classification

generate_polynomial
-------------------

.. autofunction:: generate_polynomial

Examples
--------

Regression Data
~~~~~~~~~~~~~~~

.. code-block:: python

   from jasmine.datasets import generate_regression
   import jax.numpy as jnp
   
   # Basic regression dataset
   X, y = generate_regression(
       n_samples=1000,
       n_features=20,
       n_informative=5,
       noise=0.1,
       random_state=42
   )
   
   print(f"Data shape: {X.shape}")
   print(f"Target range: [{jnp.min(y):.2f}, {jnp.max(y):.2f}]")

   # With ground truth coefficients
   X, y, coef = generate_regression(
       n_samples=500,
       n_features=10,
       coef=True,
       random_state=42
   )
   
   print(f"True coefficients: {coef}")

Classification Data
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jasmine.datasets import generate_classification
   
   # Binary classification
   X, y = generate_classification(
       n_samples=1000,
       n_features=20,
       n_informative=10,
       n_classes=2,
       class_sep=1.0,
       random_state=42
   )
   
   print(f"Data shape: {X.shape}")
   print(f"Class distribution: {jnp.bincount(y)}")

   # Multi-class classification
   X, y = generate_classification(
       n_samples=1500,
       n_features=15,
       n_classes=3,
       n_informative=8,
       random_state=42
   )

Polynomial Data
~~~~~~~~~~~~~~~

.. code-block:: python

   from jasmine.datasets import generate_polynomial
   
   # Polynomial regression dataset
   X, y = generate_polynomial(
       n_samples=500,
       degree=3,
       noise=0.1,
       random_state=42
   )
   
   print(f"Polynomial data shape: {X.shape}")

Usage Tips
~~~~~~~~~~

* Use ``random_state`` for reproducible datasets
* Adjust ``noise`` parameter to control data difficulty
* ``n_informative`` controls how many features actually affect the target
* ``class_sep`` in classification controls how well-separated the classes are
* All generators return JAX arrays for seamless integration
