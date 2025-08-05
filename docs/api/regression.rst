Regression Models
================

The :mod:`jasmine.regression` module provides linear regression models with JAX acceleration.

.. currentmodule:: jasmine.regression

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   LinearRegression

LinearRegression
----------------

.. autoclass:: LinearRegression
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::
      :toctree: generated/

      ~LinearRegression.__init__
      ~LinearRegression.train
      ~LinearRegression.inference
      ~LinearRegression.evaluate

Examples
--------

Basic Linear Regression
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jasmine.regression import LinearRegression
   from jasmine.datasets import generate_regression
   
   # Generate synthetic data
   X, y = generate_regression(n_samples=1000, n_features=10, noise=0.1)
   
   # Create and train model
   model = LinearRegression(learning_rate=0.01, n_epochs=1000)
   model.train(X, y)
   
   # Make predictions
   predictions = model.inference(X)
   r2_score = model.evaluate(X, y)
   print(f"R² Score: {r2_score:.4f}")

Linear Regression with Regularization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # L2 regularization (Ridge)
   model_ridge = LinearRegression(
       learning_rate=0.01,
       n_epochs=2000,
       l2_penalty=0.1
   )
   
   # L1 regularization (Lasso)
   model_lasso = LinearRegression(
       learning_rate=0.01,
       n_epochs=2000,
       l1_penalty=0.1
   )
   
   # Elastic Net (L1 + L2)
   model_elastic = LinearRegression(
       learning_rate=0.01,
       n_epochs=2000,
       l1_penalty=0.05,
       l2_penalty=0.05
   )

Performance Tips
~~~~~~~~~~~~~~~~

* Use feature scaling with :class:`~jasmine.preprocessing.StandardScaler` for better convergence
* Larger learning rates work well with JIT compilation
* First run includes compilation overhead (~1-3s), subsequent runs are fast
* GPU acceleration is automatic when JAX detects CUDA
