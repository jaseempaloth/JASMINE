Quick Start Guide
=================

This guide will get you up and running with JASMINE in just a few minutes.

Installation
------------

Install JASMINE from source:

.. code-block:: bash

   git clone https://github.com/jaseempaloth/JASMINE.git
   cd JASMINE
   pip install -r requirements.txt
   pip install -e .

Verify the installation:

.. code-block:: python

   import jasmine
   print(f"JASMINE version: {jasmine.__version__}")

Your First Model
----------------

Linear Regression
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jasmine.regression import LinearRegression
   from jasmine.datasets import generate_regression
   from jasmine.selection import train_test_split

   # Generate synthetic data
   X, y = generate_regression(n_samples=1000, n_features=10, noise=0.1)

   # Split data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

   # Create and train model
   model = LinearRegression(learning_rate=0.01, n_epochs=1000)
   model.train(X_train, y_train)

   # Evaluate
   r2_score = model.evaluate(X_test, y_test)
   print(f"R² Score: {r2_score:.4f}")

Logistic Regression
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jasmine.classification import LogisticRegression
   from jasmine.datasets import generate_classification

   # Generate classification data
   X, y = generate_classification(n_samples=1000, n_features=10, n_classes=2)

   # Split data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

   # Create and train model
   model = LogisticRegression(learning_rate=0.1, n_epochs=1000)
   model.train(X_train, y_train)

   # Make predictions
   predictions = model.inference(X_test)
   probabilities = model.predict_probabilities(X_test)
   accuracy = model.evaluate(X_test, y_test)
   
   print(f"Accuracy: {accuracy:.4f}")

Data Preprocessing
------------------

Feature Scaling
~~~~~~~~~~~~~~~

.. code-block:: python

   from jasmine.preprocessing import StandardScaler

   # Create scaler
   scaler = StandardScaler()

   # Fit on training data and transform
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)

   # Train on scaled data
   model = LinearRegression()
   model.train(X_train_scaled, y_train)

Complete Pipeline
-----------------

Here's a complete example combining all components:

.. code-block:: python

   import jax.numpy as jnp
   from jasmine.regression import LinearRegression
   from jasmine.preprocessing import StandardScaler
   from jasmine.datasets import generate_regression
   from jasmine.selection import train_test_split
   from jasmine.metrics import mean_squared_error

   # 1. Generate data
   X, y = generate_regression(
       n_samples=2000, 
       n_features=15, 
       noise=0.1, 
       random_state=42
   )

   # 2. Split data
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   # 3. Scale features
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)

   # 4. Train model
   model = LinearRegression(
       learning_rate=0.01,
       n_epochs=2000,
       l2_penalty=0.01  # Add regularization
   )
   model.train(X_train_scaled, y_train)

   # 5. Evaluate
   r2 = model.evaluate(X_test_scaled, y_test)
   mse = model.evaluate(X_test_scaled, y_test, metrics_fn=mean_squared_error)

   print(f"R² Score: {r2:.4f}")
   print(f"MSE: {mse:.4f}")

   # 6. Make predictions
   predictions = model.inference(X_test_scaled)
   print(f"Predictions shape: {predictions.shape}")

Key Concepts
------------

JIT Compilation
~~~~~~~~~~~~~~~

JASMINE uses JAX's Just-In-Time compilation for speed:

* **First run**: Includes compilation overhead (~1-3 seconds)
* **Subsequent runs**: Native speed, often 10-100x faster than NumPy

.. code-block:: python

   import time

   # First training (with compilation)
   start = time.time()
   model.train(X_train, y_train)
   first_time = time.time() - start

   # Second training (compiled)
   start = time.time()
   model.train(X_train, y_train)  
   second_time = time.time() - start

   print(f"First run: {first_time:.3f}s")
   print(f"Second run: {second_time:.3f}s")
   print(f"Speedup: {first_time/second_time:.1f}x")

GPU Acceleration
~~~~~~~~~~~~~~~~

JAX automatically uses GPU when available:

.. code-block:: python

   import jax
   
   # Check available devices
   print("Available devices:", jax.devices())
   
   # Training automatically uses GPU if available
   model.train(X_train, y_train)  # Runs on GPU if CUDA is available

Next Steps
----------

* Read the :doc:`examples` for detailed tutorials
* Explore the :doc:`api/index` for complete function reference
* Check :doc:`performance` for optimization tips
* See :doc:`installation` for advanced setup options
