# JASMINE - JAX Accelerated Statistical Models and Integrated Neural Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-latest-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

JASMINE is a high-performance machine learning library built on JAX, designed to leverage GPU and TPU acceleration for statistical modeling and neural computation.

## Features

- **JIT-compiled models**: Linear/Logistic Regression with GPU/TPU acceleration
- **Multiple optimizers**: SGD, Momentum, Adam with adaptive learning rates  
- **Advanced regularization**: L1, L2, Elastic Net penalties
- **Data preprocessing**: StandardScaler with JIT acceleration
- **Sklearn-compatible API**: Familiar interface with JAX performance
- **Automatic differentiation**: Powered by JAX's grad transformations

## Quick Start

```python
from jasmine import LinearRegression, LogisticRegression
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

- **[Quick Start Guide](docs/quickstart.md)** - Get up and running in 5 minutes
- **[API Reference](docs/api.md)** - Complete function and class documentation  
- **[Examples](docs/examples.md)** - Detailed tutorials and use cases
- **[Performance Guide](docs/performance.md)** - Benchmarks and optimization tips
- **[Installation Guide](docs/installation.md)** - Platform-specific setup instructions
- **[FAQ](docs/faq.md)** - Common questions and troubleshooting

## Requirements

- Python 3.8+
- JAX >= 0.4.0
- NumPy >= 1.21.0

## License

MIT License - see [LICENSE](LICENSE) file for details.- JAX Accelerated Statistical Models and Integrated Neural Engine

JASMINE is a lightweight machine learning library built on top of JAX, designed to leverage GPU and TPU acceleration for high-performance computing. The project aims to provide an accessible interface to JAX’s powerful capabilities while continuously updating with new features and models.

## Features

- Modular and extensible framework supporting diverse machine learning algorithms
- Core components for model development, training, and evaluation
- Data preprocessing utilities
- Automatic differentiation powered by JAX

