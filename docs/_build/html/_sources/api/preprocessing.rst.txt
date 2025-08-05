Data Preprocessing
==================

The :mod:`jasmine.preprocessing` module provides data preprocessing utilities.

.. currentmodule:: jasmine.preprocessing

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   StandardScaler

StandardScaler
--------------

.. autoclass:: StandardScaler
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::
      :toctree: generated/

      ~StandardScaler.__init__
      ~StandardScaler.fit
      ~StandardScaler.transform
      ~StandardScaler.fit_transform
      ~StandardScaler.inverse_transform

   .. rubric:: Properties

   .. autosummary::
      :toctree: generated/

      ~StandardScaler.is_fitted

Examples
--------

Basic Feature Scaling
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jasmine.preprocessing import StandardScaler
   import jax.numpy as jnp
   
   # Create sample data with different scales
   X = jnp.array([[1, 100, 10000],
                  [2, 200, 20000],
                  [3, 300, 30000]])
   
   # Fit and transform
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   
   print("Original data shape:", X.shape)
   print("Scaled data mean:", jnp.mean(X_scaled, axis=0))
   print("Scaled data std:", jnp.std(X_scaled, axis=0))

Preprocessing Pipeline
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jasmine.preprocessing import StandardScaler
   from jasmine.regression import LinearRegression
   from jasmine.selection import train_test_split
   
   # Split data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   
   # Fit scaler on training data only
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   
   # Train model on scaled data
   model = LinearRegression()
   model.train(X_train_scaled, y_train)
   
   # Evaluate
   score = model.evaluate(X_test_scaled, y_test)
   print(f"R² Score: {score:.4f}")

Inverse Transformation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Transform data
   X_scaled = scaler.fit_transform(X)
   
   # Recover original data
   X_recovered = scaler.inverse_transform(X_scaled)
   
   # Verify recovery
   recovery_error = jnp.mean(jnp.abs(X - X_recovered))
   print(f"Recovery error: {recovery_error:.2e}")

Performance Notes
~~~~~~~~~~~~~~~~~

* StandardScaler uses JIT compilation for fast transforms
* ``epsilon`` parameter prevents division by zero for constant features
* Scaling parameters are stored in ``params`` dictionary
* Use ``is_fitted`` property to check if scaler has been fitted
