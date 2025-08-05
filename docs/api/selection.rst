Data Selection
==============

The :mod:`jasmine.selection` module provides data splitting utilities.

.. currentmodule:: jasmine.selection

Functions
---------

.. autosummary::
   :toctree: generated/

   train_test_split

train_test_split
----------------

.. autofunction:: train_test_split

Examples
--------

Basic Data Splitting
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jasmine.selection import train_test_split
   from jasmine.datasets import generate_regression
   
   # Generate sample data
   X, y = generate_regression(n_samples=1000, n_features=10)
   
   # Split into train and test sets
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, 
       test_size=0.2, 
       random_state=42
   )
   
   print(f"Training set: {X_train.shape[0]} samples")
   print(f"Test set: {X_test.shape[0]} samples")

Custom Split Ratios
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 80-20 split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2
   )
   
   # 70-30 split  
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3
   )
   
   # 90-10 split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.1
   )

Reproducible Splits
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Same split every time
   X_train1, X_test1, y_train1, y_test1 = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   
   X_train2, X_test2, y_train2, y_test2 = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   
   # These will be identical
   assert jnp.array_equal(X_train1, X_train2)
   assert jnp.array_equal(y_train1, y_train2)

No Shuffling
~~~~~~~~~~~~

.. code-block:: python

   # Keep original order
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, 
       test_size=0.2, 
       shuffle=False
   )
   
   # First 80% will be training, last 20% will be test

Three-Way Split
~~~~~~~~~~~~~~~

.. code-block:: python

   # Split into train, validation, and test
   X_temp, X_test, y_temp, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   
   X_train, X_val, y_train, y_val = train_test_split(
       X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
   )
   
   print(f"Train: {X_train.shape[0]} samples (60%)")
   print(f"Validation: {X_val.shape[0]} samples (20%)")
   print(f"Test: {X_test.shape[0]} samples (20%)")

Function Properties
~~~~~~~~~~~~~~~~~~~

* Preserves JAX array types
* Supports any test_size between 0.0 and 1.0
* Guarantees at least 1 sample in test set
* Efficient implementation using JAX operations
* Compatible with all JASMINE data generators
