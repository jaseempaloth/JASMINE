API Reference
=============

This section contains the complete API reference for JASMINE.

Core Modules
------------

.. toctree::
   :maxdepth: 2

   regression
   classification
   preprocessing
   datasets
   metrics
   selection

Package Overview
----------------

JASMINE is organized into several modules, each providing specific functionality:

* :mod:`jasmine.regression` - Linear regression models
* :mod:`jasmine.classification` - Classification models  
* :mod:`jasmine.preprocessing` - Data preprocessing utilities
* :mod:`jasmine.datasets` - Synthetic data generators
* :mod:`jasmine.metrics` - Performance metrics
* :mod:`jasmine.selection` - Data splitting utilities

Quick Reference
---------------

Most Common Classes
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   jasmine.regression.LinearRegression
   jasmine.classification.LogisticRegression
   jasmine.preprocessing.StandardScaler

Most Common Functions
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   jasmine.datasets.generate_regression
   jasmine.datasets.generate_classification
   jasmine.selection.train_test_split
   jasmine.metrics.mean_squared_error
   jasmine.metrics.accuracy_score
