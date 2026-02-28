Quick Start
===========

Install dependencies and package:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -e .

Train a regression model:

.. code-block:: python

   from jasmine.linear_model import LinearRegression
   from jasmine.datasets import generate_regression

   X, y = generate_regression(n_samples=1000, n_features=20, random_state=42)
   model = LinearRegression(learning_rate=0.01, n_epochs=1000)
   model.train(X, y)
   preds = model.inference(X)
   score = model.evaluate(X, y)
   print(score)
