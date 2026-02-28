# JASMINE - JAX Accelerated Statistical Models and Integrated Neural Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-latest-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-jasmine-brightgreen.svg)](https://jasmine-ml.readthedocs.io)

JASMINE is a lightweight machine learning library built on top of JAX, designed to leverage GPU and TPU acceleration while maintaining a minimal and transparent architecture. The project focuses on clear, modular reference implementations of core machine learning algorithms, with emphasis on readability, inspectability, and algorithmic understanding.

📖 **[Read the Documentation](https://jasmine-ml.readthedocs.io)** | 🚀 **[Quick Start](https://jasmine-ml.readthedocs.io/en/latest/quickstart.html)** | 📚 **[API Reference](https://jasmine-ml.readthedocs.io/en/latest/api/)**

## Core Concepts

- **JIT-compiled models**: Linear/Logistic Regression with GPU/TPU acceleration
- **Multiple optimizers**: SGD, Momentum, Adam with adaptive learning rates  
- **Advanced regularization**: L1, L2, Elastic Net penalties
- **Data preprocessing**: StandardScaler with JIT acceleration
- **Sklearn-inspired API**: Familiar interface while keeping internals easy to follow
- **Automatic differentiation**: Powered by JAX's grad transformations

## Design Focus

- **Readable internals**: Small modules and straightforward training loops
- **Experiment-friendly workflow**: Easy to modify, test, and compare algorithm variants
- **Strong foundations**: Prioritizes core methods and clean implementations over broad feature coverage

## Quick Start

```python
from jasmine.linear_model import LogisticRegression
from jasmine.datasets import generate_classification

# Generate data and train model
X, y = generate_classification(n_samples=1000, n_features=20)
model = LogisticRegression(learning_rate=0.1, n_epochs=1000)
model.train(X, y)

# Make predictions
predictions = model.inference(X)
accuracy = model.evaluate(X, y)
print(f"Accuracy: {accuracy:.3f}")
```

## Installation

```bash
git clone https://github.com/jaseempaloth/JASMINE.git
cd JASMINE
pip install -r requirements.txt
pip install -e .
```

## Documentation

📖 **Complete documentation is available at [jasmine.readthedocs.io](https://jasmine-ml.readthedocs.io)**

**Quick Links:**
- 🚀 [Quick Start Guide](https://jasmine-ml.readthedocs.io/en/latest/quickstart.html) - Get up and running in 5 minutes
- 📚 [API Reference](https://jasmine-ml.readthedocs.io/en/latest/api/) - Complete function and class documentation  
- 📝 [Examples & Tutorials](https://jasmine-ml.readthedocs.io/en/latest/examples.html) - Detailed use cases and best practices
- 💾 [Installation Guide](https://jasmine-ml.readthedocs.io/en/latest/installation.html) - Platform-specific setup instructions

### Rebuild docs locally

```bash
pip install -r docs/requirements.txt
pip install -e .[dev]
sphinx-build -b html -E -a docs docs/_build/html
```

Build output is written to `docs/_build/html/` (open `docs/_build/html/index.html`).

Notes:
- In offline environments, intersphinx inventory fetch warnings are expected.
- The current docs may emit duplicate autodoc object warnings, but HTML output is still generated.

## Requirements

- Python 3.8+
- JAX >= 0.4.0

## License

MIT License - see [LICENSE](LICENSE) file for details.
