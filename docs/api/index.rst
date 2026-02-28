API Reference
=============

This section contains the complete API reference for JASMINE.

Core Modules
------------

.. toctree::
   :maxdepth: 2

   linear_model
   neighbors
   svm
   preprocessing
   datasets
   metrics
   model_selection

Package Overview
----------------

JASMINE is organized into several modules, each providing specific functionality:

* :mod:`jasmine.linear_model` - Linear and logistic regression models
* :mod:`jasmine.neighbors` - Nearest-neighbor classifiers
* :mod:`jasmine.svm` - Support Vector Machine classifiers
* :mod:`jasmine.preprocessing` - Data preprocessing utilities
* :mod:`jasmine.datasets` - Synthetic data generators
* :mod:`jasmine.metrics` - Performance metrics
* :mod:`jasmine.model_selection` - Data splitting utilities

Quick Reference
---------------

Most Common Classes
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   jasmine.linear_model.LinearRegression
   jasmine.linear_model.LogisticRegression
   jasmine.neighbors.KNNClassifier
   jasmine.svm.SVMClassifier
   jasmine.preprocessing.StandardScaler

Most Common Functions
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   jasmine.datasets.generate_regression
   jasmine.datasets.generate_classification
   jasmine.model_selection.train_test_split
   jasmine.metrics.mean_squared_error
   jasmine.metrics.accuracy_score
