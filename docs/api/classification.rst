Classification Models
====================

The :mod:`jasmine.classification` module provides classification models with JAX acceleration.

.. currentmodule:: jasmine.classification

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   LogisticRegression

LogisticRegression
------------------

.. autoclass:: LogisticRegression
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::
      :toctree: generated/

      ~LogisticRegression.__init__
      ~LogisticRegression.train
      ~LogisticRegression.inference
      ~LogisticRegression.predict_probabilities
      ~LogisticRegression.evaluate

Examples
--------

Binary Classification
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from jasmine.classification import LogisticRegression
   from jasmine.datasets import generate_classification
   from jasmine.metrics import accuracy_score
   
   # Generate synthetic data
   X, y = generate_classification(n_samples=1000, n_features=10, n_classes=2)
   
   # Create and train model
   model = LogisticRegression(learning_rate=0.1, n_epochs=1000)
   model.train(X, y)
   
   # Make predictions
   predictions = model.inference(X)
   probabilities = model.predict_probabilities(X)
   accuracy = model.evaluate(X, y, metrics_fn=accuracy_score)
   
   print(f"Accuracy: {accuracy:.4f}")

Classification with Regularization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # L2 regularization for overfitting prevention
   model = LogisticRegression(
       learning_rate=0.1,
       n_epochs=2000,
       l2_penalty=0.01
   )
   
   model.train(X, y)
   
   # Evaluate model confidence
   probabilities = model.predict_probabilities(X)
   high_confidence = jnp.sum((probabilities > 0.8) | (probabilities < 0.2))
   print(f"High confidence predictions: {high_confidence}/{len(probabilities)}")

Performance Tips
~~~~~~~~~~~~~~~~

* Use feature scaling for better convergence
* Monitor training with ``verbose=1`` parameter
* L2 regularization helps with high-dimensional data
* JAX automatically uses GPU when available
