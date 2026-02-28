Examples & Tutorials
====================

Complete examples demonstrating JASMINE's capabilities.

Basic Examples
--------------

Example 1: Linear Regression with Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import jax.numpy as jnp
   from jasmine.linear_model import LinearRegression
   from jasmine.preprocessing import StandardScaler
   from jasmine.datasets import generate_regression
   from jasmine.model_selection import train_test_split
   import matplotlib.pyplot as plt

   # Generate realistic dataset
   X, y = generate_regression(
       n_samples=5000,
       n_features=20, 
       noise=0.2,
       random_state=42
   )

   # Multiple train-test splits for cross-validation
   scores = []
   for seed in range(5):
       # Split data
       X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.2, random_state=seed
       )
       
       # Preprocess
       scaler = StandardScaler()
       X_train_scaled = scaler.fit_transform(X_train)
       X_test_scaled = scaler.transform(X_test)
       
       # Train with regularization
       model = LinearRegression(
           learning_rate=0.01,
           n_epochs=3000,
           l2_penalty=0.1
       )
       model.train(X_train_scaled, y_train)
       
       # Evaluate
       score = model.evaluate(X_test_scaled, y_test)
       scores.append(score)
       print(f"Fold {seed + 1}: R² = {score:.4f}")

   print(f"\\nMean R²: {jnp.mean(jnp.array(scores)):.4f} ± {jnp.std(jnp.array(scores)):.4f}")

Example 2: Multi-class Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jasmine.linear_model import LogisticRegression
   from jasmine.datasets import generate_classification
   from jasmine.metrics import accuracy_score
   import jax.numpy as jnp

   # Generate multi-class data
   X, y = generate_classification(
       n_samples=3000,
       n_features=10,
       n_classes=4,  # 4-class problem
       random_state=42
   )

   # Split data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

   # Scale features
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)

   # Train classifier
   model = LogisticRegression(
       learning_rate=0.1,
       n_epochs=2000,
       l2_penalty=0.01
   )
   model.train(X_train_scaled, y_train)

   # Detailed evaluation
   predictions = model.inference(X_test_scaled)
   probabilities = model.predict_probabilities(X_test_scaled)

   # Calculate accuracy for each class
   for class_idx in range(4):
       class_mask = y_test == class_idx
       if jnp.sum(class_mask) > 0:
           class_acc = jnp.mean(predictions[class_mask] == class_idx)
           print(f"Class {class_idx} accuracy: {class_acc:.4f}")

   overall_acc = accuracy_score(y_test, predictions)
   print(f"Overall accuracy: {overall_acc:.4f}")

Advanced Examples
-----------------

Example 3: Regularization Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   from jasmine.linear_model import LinearRegression
   from jasmine.metrics import mean_squared_error

   # Generate data with multicollinearity
   X, y = generate_regression(
       n_samples=1000,
       n_features=50,  # High dimensional
       noise=0.3,
       random_state=42
   )

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

   # Scale data
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)

   # Test different regularization strengths
   penalties = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
   train_errors = []
   test_errors = []

   for penalty in penalties:
       model = LinearRegression(
           learning_rate=0.01,
           n_epochs=2000,
           l2_penalty=penalty
       )
       model.train(X_train_scaled, y_train)
       
       # Calculate errors
       train_pred = model.inference(X_train_scaled)
       test_pred = model.inference(X_test_scaled)
       
       train_mse = mean_squared_error(y_train, train_pred)
       test_mse = mean_squared_error(y_test, test_pred)
       
       train_errors.append(train_mse)
       test_errors.append(test_mse)
       
       print(f"L2={penalty:.3f}: Train MSE={train_mse:.4f}, Test MSE={test_mse:.4f}")

   # Plot results
   plt.figure(figsize=(10, 6))
   plt.semilogx(penalties, train_errors, 'o-', label='Training Error')
   plt.semilogx(penalties, test_errors, 's-', label='Test Error')
   plt.xlabel('L2 Penalty')
   plt.ylabel('Mean Squared Error')
   plt.title('Regularization Effect on Linear Regression')
   plt.legend()
   plt.grid(True)
   plt.show()

Example 4: Learning Curve Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def plot_learning_curve(model_class, X, y, **model_kwargs):
       """Plot learning curve showing performance vs training set size."""
       
       # Different training set sizes
       train_sizes = jnp.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
       train_scores = []
       val_scores = []
       
       # Fixed validation set
       X_temp, X_val, y_temp, y_val = train_test_split(X, y, test_size=0.2)
       
       for size in train_sizes:
           # Variable training set size
           n_train = int(size * len(X_temp))
           X_train = X_temp[:n_train]
           y_train = y_temp[:n_train]
           
           # Scale data
           scaler = StandardScaler()
           X_train_scaled = scaler.fit_transform(X_train)
           X_val_scaled = scaler.transform(X_val)
           
           # Train model
           model = model_class(**model_kwargs)
           model.train(X_train_scaled, y_train)
           
           # Evaluate
           train_score = model.evaluate(X_train_scaled, y_train)
           val_score = model.evaluate(X_val_scaled, y_val)
           
           train_scores.append(train_score)
           val_scores.append(val_score)
           
           print(f"Size {size:.1f}: Train={train_score:.4f}, Val={val_score:.4f}")
       
       # Plot
       plt.figure(figsize=(10, 6))
       plt.plot(train_sizes, train_scores, 'o-', label='Training Score')
       plt.plot(train_sizes, val_scores, 's-', label='Validation Score')
       plt.xlabel('Training Set Size (fraction)')
       plt.ylabel('R² Score')
       plt.title('Learning Curve')
       plt.legend()
       plt.grid(True)
       plt.show()

   # Usage
   X, y = generate_regression(n_samples=2000, n_features=15, noise=0.1)
   plot_learning_curve(
       LinearRegression, 
       X, y,
       learning_rate=0.01,
       n_epochs=2000,
       l2_penalty=0.01
   )

Performance Examples
--------------------

Example 5: GPU vs CPU Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   import jax

   def benchmark_training(X, y, device='cpu'):
       """Benchmark training time on different devices."""
       
       # Move data to device
       with jax.default_device(jax.devices(device)[0]):
           X_device = jax.device_put(X)
           y_device = jax.device_put(y)
           
           model = LinearRegression(learning_rate=0.01, n_epochs=1000)
           
           # Warm-up (compile)
           model.train(X_device[:100], y_device[:100])
           
           # Actual benchmark
           start_time = time.time()
           model.train(X_device, y_device)
           training_time = time.time() - start_time
           
       return training_time

   # Generate large dataset
   X, y = generate_regression(n_samples=10000, n_features=100, noise=0.1)

   # Benchmark CPU
   cpu_time = benchmark_training(X, y, 'cpu')
   print(f"CPU training time: {cpu_time:.3f} seconds")

   # Benchmark GPU (if available)
   if jax.devices('gpu'):
       gpu_time = benchmark_training(X, y, 'gpu') 
       print(f"GPU training time: {gpu_time:.3f} seconds")
       print(f"GPU speedup: {cpu_time/gpu_time:.1f}x")
   else:
       print("GPU not available")

Example 6: Large-Scale Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def large_scale_pipeline():
       """Demonstrate JASMINE on a large dataset."""
       
       print("Generating large dataset...")
       X, y = generate_regression(
           n_samples=50000,  # Large dataset
           n_features=200,   # High-dimensional
           noise=0.1,
           random_state=42
       )
       print(f"Dataset shape: {X.shape}")
       
       # Split data
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
       
       # Preprocessing
       print("Scaling features...")
       scaler = StandardScaler()
       X_train_scaled = scaler.fit_transform(X_train)
       X_test_scaled = scaler.transform(X_test)
       
       # Training with progress tracking
       print("Training model...")
       model = LinearRegression(
           learning_rate=0.001,  # Lower LR for stability
           n_epochs=5000,
           l2_penalty=0.1
       )
       
       start_time = time.time()
       model.train(X_train_scaled, y_train)
       training_time = time.time() - start_time
       
       # Evaluation
       print("Evaluating...")
       train_score = model.evaluate(X_train_scaled, y_train)
       test_score = model.evaluate(X_test_scaled, y_test)
       
       print(f"\\nResults:")
       print(f"Training time: {training_time:.2f} seconds")
       print(f"Training R²: {train_score:.4f}")
       print(f"Test R²: {test_score:.4f}")
       print(f"Samples per second: {len(X_train)/training_time:.0f}")

   # Run large-scale example
   large_scale_pipeline()

Tips and Best Practices
------------------------

Memory Management
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For very large datasets, use batch processing
   def batch_inference(model, X, batch_size=1000):
       """Process large datasets in batches."""
       n_samples = X.shape[0]
       predictions = []
       
       for start_idx in range(0, n_samples, batch_size):
           end_idx = min(start_idx + batch_size, n_samples)
           batch_X = X[start_idx:end_idx]
           batch_pred = model.inference(batch_X)
           predictions.append(batch_pred)
           
       return jnp.concatenate(predictions)

Hyperparameter Tuning
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Grid search for optimal hyperparameters
   def grid_search_cv(X, y, param_grid, cv_folds=5):
       """Simple grid search with cross-validation."""
       best_score = -jnp.inf
       best_params = None
       
       for lr in param_grid['learning_rate']:
           for penalty in param_grid['l2_penalty']:
               scores = []
               
               for fold in range(cv_folds):
                   X_train, X_val, y_train, y_val = train_test_split(
                       X, y, test_size=0.2, random_state=fold
                   )
                   
                   scaler = StandardScaler()
                   X_train_scaled = scaler.fit_transform(X_train)
                   X_val_scaled = scaler.transform(X_val)
                   
                   model = LinearRegression(
                       learning_rate=lr,
                       n_epochs=2000,
                       l2_penalty=penalty
                   )
                   model.train(X_train_scaled, y_train)
                   score = model.evaluate(X_val_scaled, y_val)
                   scores.append(score)
               
               mean_score = jnp.mean(jnp.array(scores))
               
               if mean_score > best_score:
                   best_score = mean_score
                   best_params = {'learning_rate': lr, 'l2_penalty': penalty}
       
       return best_params, best_score

   # Usage
   param_grid = {
       'learning_rate': [0.001, 0.01, 0.1],
       'l2_penalty': [0.0, 0.01, 0.1, 1.0]
   }
   
   best_params, best_score = grid_search_cv(X, y, param_grid)
   print(f"Best parameters: {best_params}")
   print(f"Best CV score: {best_score:.4f}")

For more examples, see the ``examples/`` directory in the repository.
