Performance Metrics
===================

The :mod:`jasmine.metrics` module provides performance evaluation metrics.

.. currentmodule:: jasmine.metrics

Regression Metrics
------------------

.. autosummary::
   :toctree: generated/

   mean_squared_error
   mean_absolute_error
   root_mean_squared_error
   r2_score

Classification Metrics
----------------------

.. autosummary::
   :toctree: generated/

   accuracy_score
   binary_cross_entropy
   categorical_cross_entropy

Regression Functions
--------------------

mean_squared_error
~~~~~~~~~~~~~~~~~~

.. autofunction:: mean_squared_error

mean_absolute_error
~~~~~~~~~~~~~~~~~~~

.. autofunction:: mean_absolute_error

root_mean_squared_error
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: root_mean_squared_error

r2_score
~~~~~~~~

.. autofunction:: r2_score

Classification Functions
------------------------

accuracy_score
~~~~~~~~~~~~~~

.. autofunction:: accuracy_score

binary_cross_entropy
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: binary_cross_entropy

categorical_cross_entropy
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: categorical_cross_entropy

Examples
--------

Regression Metrics
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jasmine.metrics import (
       mean_squared_error, 
       mean_absolute_error, 
       r2_score
   )
   import jax.numpy as jnp
   
   # Sample predictions and targets
   y_true = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
   y_pred = jnp.array([1.1, 1.9, 3.2, 3.8, 5.1])
   
   # Calculate metrics
   mse = mean_squared_error(y_true, y_pred)
   mae = mean_absolute_error(y_true, y_pred)
   r2 = r2_score(y_true, y_pred)
   
   print(f"MSE: {mse:.4f}")
   print(f"MAE: {mae:.4f}")
   print(f"R²: {r2:.4f}")

Classification Metrics
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jasmine.metrics import accuracy_score, binary_cross_entropy
   import jax.numpy as jnp
   
   # Binary classification
   y_true = jnp.array([0, 1, 1, 0, 1])
   y_pred_classes = jnp.array([0, 1, 1, 0, 0])
   y_pred_probs = jnp.array([0.1, 0.9, 0.8, 0.2, 0.4])
   
   # Calculate metrics
   accuracy = accuracy_score(y_true, y_pred_classes)
   bce = binary_cross_entropy(y_true, y_pred_probs)
   
   print(f"Accuracy: {accuracy:.4f}")
   print(f"Binary Cross-Entropy: {bce:.4f}")

Using with Models
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jasmine.regression import LinearRegression
   from jasmine.metrics import mean_squared_error
   
   # Train model
   model = LinearRegression()
   model.train(X_train, y_train)
   
   # Evaluate with custom metric
   mse = model.evaluate(X_test, y_test, metrics_fn=mean_squared_error)
   r2 = model.evaluate(X_test, y_test)  # Default metric
   
   print(f"MSE: {mse:.4f}")
   print(f"R²: {r2:.4f}")

Metric Properties
~~~~~~~~~~~~~~~~~

* All metrics are JIT-compiled for fast computation
* Functions accept JAX arrays as input
* Binary cross-entropy supports both probabilities and logits
* Metrics are designed to work seamlessly with JASMINE models
