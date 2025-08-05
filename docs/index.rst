Welcome to JASMINE's Documentation!
=====================================

**JASMINE** (JAX Accelerated Statistical Models and Integrated Neural Engine) is a high-performance machine learning library built on JAX, designed to leverage GPU and TPU acceleration for statistical modeling and neural computation.

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/JAX-latest-orange.svg
   :target: https://github.com/google/jax
   :alt: JAX

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Key Features
------------

* **JIT-compiled models**: Linear/Logistic Regression with GPU/TPU acceleration
* **Multiple optimizers**: SGD, Momentum, Adam with adaptive learning rates  
* **Advanced regularization**: L1, L2, Elastic Net penalties
* **Data preprocessing**: StandardScaler with JIT acceleration
* **Sklearn-compatible API**: Familiar interface with JAX performance
* **Automatic differentiation**: Powered by JAX's grad transformations

Quick Start
-----------

.. code-block:: python

   from jasmine.regression import LinearRegression
   from jasmine.datasets import generate_regression

   # Generate data and train model
   X, y = generate_regression(n_samples=1000, n_features=20)
   model = LinearRegression(learning_rate=0.1, n_epochs=1000)
   model.train(X, y)

   # Make predictions
   predictions = model.inference(X)
   r2_score = model.evaluate(X, y)
   print(f"R² Score: {r2_score:.3f}")

Installation
------------

Install JASMINE from source:

.. code-block:: bash

   git clone https://github.com/jaseempaloth/JASMINE.git
   cd JASMINE
   pip install -r requirements.txt
   pip install -e .

Optional dependencies:

.. code-block:: bash

   # Development tools
   pip install jasmine[dev]

   # Example dependencies
   pip install jasmine[examples]

   # Everything
   pip install jasmine[all]

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   installation
   examples

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/index
   api/regression
   api/classification
   api/preprocessing
   api/datasets
   api/metrics
   api/selection

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   performance
   faq
   changelog
   contributing

Performance
-----------

JASMINE leverages JAX's JIT compilation for significant performance improvements:

* **10-100x speedup** vs pure NumPy implementations
* **GPU/TPU acceleration** for large-scale datasets
* **Memory efficient** operations with automatic optimization
* **First-run compilation overhead** (~1-3s), then native speed

Requirements
------------

* Python 3.8 or higher
* JAX >= 0.4.0
* JAXlib >= 0.4.0

License
-------

This project is licensed under the MIT License - see the `LICENSE <https://github.com/jaseempaloth/JASMINE/blob/main/LICENSE>`_ file for details.

Contributing
------------

We welcome contributions! Please see our :doc:`contributing` guide for details on how to get started.

Support
-------

* **GitHub Issues**: `Report bugs and request features <https://github.com/jaseempaloth/JASMINE/issues>`_
* **Documentation**: Complete guides and API reference
* **Examples**: Real-world use cases and tutorials

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
