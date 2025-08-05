Installation Guide
==================

System Requirements
-------------------

JASMINE requires:

* **Python**: 3.7 or higher (3.8+ recommended)
* **Operating System**: Linux, macOS, or Windows
* **Memory**: 4GB RAM minimum (8GB+ recommended for large datasets)
* **Disk Space**: 1GB free space

Core Dependencies
-----------------

JASMINE's core dependencies are automatically installed:

* **JAX** (>=0.4.0): High-performance numerical computing
* **JAXlib**: JAX's XLA backend
* **NumPy**: Numerical arrays and operations

Optional dependencies are available via extras groups:

* **Development tools**: pytest, black, sphinx
* **Examples**: matplotlib, jupyter, pandas
* **All extras**: Complete development environment

Installation Methods
---------------------

Method 1: From Source (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/jaseempaloth/JASMINE.git
   cd JASMINE
   
   # Install core package
   pip install -e .
   
   # Or install with development tools
   pip install -e ".[dev]"
   
   # Or install with examples dependencies
   pip install -e ".[examples]"
   
   # Or install everything
   pip install -e ".[all]"

Method 2: Direct Install
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install core dependencies first
   pip install "jax>=0.4.0" numpy
   
   # Install JASMINE
   pip install /path/to/JASMINE

Method 3: Requirements File
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Using the included requirements.txt
   cd JASMINE
   pip install -r requirements.txt
   pip install -e .

GPU Support
-----------

JASMINE automatically uses GPU acceleration when available. To enable GPU support:

NVIDIA GPU (CUDA)
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install CUDA-enabled JAX
   pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   
   # Verify GPU detection
   python -c "import jax; print('GPUs:', jax.devices('gpu'))"

Google TPU
~~~~~~~~~~

.. code-block:: bash

   # Install TPU-enabled JAX  
   pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

AMD GPU (ROCm)
~~~~~~~~~~~~~~

.. code-block:: bash

   # Install ROCm-enabled JAX
   pip install "jax[rocm]" -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html

Apple Silicon (Metal)
~~~~~~~~~~~~~~~~~~~~~

JAX uses Apple's Metal Performance Shaders automatically on M1/M2 Macs:

.. code-block:: bash

   # Standard installation works on Apple Silicon
   pip install jax
   
   # Verify Metal backend
   python -c "import jax; print('Devices:', jax.devices())"

Verification
------------

Basic Installation Check
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test basic functionality
   import jasmine
   from jasmine.regression import LinearRegression
   from jasmine.datasets import generate_regression
   
   print(f"JASMINE version: {jasmine.__version__}")
   
   # Quick test
   X, y = generate_regression(n_samples=100, n_features=5)
   model = LinearRegression()
   model.train(X, y)
   score = model.evaluate(X, y)
   
   print(f"Basic test R² score: {score:.4f}")
   print("✓ Installation successful!")

GPU/TPU Verification
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import jax
   import jax.numpy as jnp
   
   # Check available devices
   print("Available devices:")
   for device in jax.devices():
       print(f"  {device}")
   
   # Test device placement
   x = jnp.array([1, 2, 3])
   print(f"Array device: {x.device()}")
   
   # Performance test
   import time
   from jasmine.regression import LinearRegression
   from jasmine.datasets import generate_regression
   
   X, y = generate_regression(n_samples=1000, n_features=50)
   model = LinearRegression(n_epochs=1000)
   
   start = time.time()
   model.train(X, y)
   duration = time.time() - start
   
   print(f"Training time: {duration:.3f} seconds")
   if jax.devices('gpu'):
       print("✓ GPU acceleration active")
   elif jax.devices('tpu'):
       print("✓ TPU acceleration active")
   else:
       print("• CPU-only mode")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Issue**: ``ImportError: No module named 'jax'``

**Solution**: Install JAX manually:

.. code-block:: bash

   pip install "jax>=0.4.0" jaxlib

**Issue**: ``RuntimeError: CUDA device not found``

**Solution**: Install CUDA-compatible JAX:

.. code-block:: bash

   pip uninstall jax jaxlib
   pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

**Issue**: Slow performance on first run

**Solution**: This is normal JAX compilation. Subsequent runs will be much faster:

.. code-block:: python

   # First run: slow (compilation)
   model.train(X, y)  # ~3 seconds
   
   # Second run: fast (compiled)
   model.train(X, y)  # ~0.1 seconds

**Issue**: Out of memory errors

**Solution**: Use smaller batch sizes or enable 64-bit precision:

.. code-block:: python

   # Enable 64-bit precision for better numerical stability
   import jax
   jax.config.update("jax_enable_x64", True)

**Issue**: ``ModuleNotFoundError: No module named 'jasmine'``

**Solution**: Install in development mode:

.. code-block:: bash

   cd JASMINE
   pip install -e .

Platform-Specific Notes
~~~~~~~~~~~~~~~~~~~~~~~

**macOS with Apple Silicon**:

.. code-block:: bash

   # May need to install specific JAX version
   pip install jax-metal
   
**Windows**:

.. code-block:: bash

   # Use conda for easier dependency management
   conda install jax -c conda-forge
   pip install -e .

**Linux with old GLIBC**:

.. code-block:: bash

   # Use manylinux wheels
   pip install --only-binary=all jax jaxlib

Development Installation
------------------------

For contributors and developers:

.. code-block:: bash

   # Clone and install in development mode
   git clone https://github.com/jaseempaloth/JASMINE.git
   cd JASMINE
   
   # Install with all development dependencies
   pip install -e ".[dev]"
   
   # Install pre-commit hooks (optional)
   pre-commit install
   
   # Run tests
   python -m pytest tests/
   
   # Build documentation
   cd docs_sphinx
   make html

Virtual Environments
--------------------

Using conda:

.. code-block:: bash

   # Create environment
   conda create -n jasmine python=3.9
   conda activate jasmine
   
   # Install JASMINE
   pip install -e ".[all]"

Using venv:

.. code-block:: bash

   # Create environment
   python -m venv jasmine_env
   source jasmine_env/bin/activate  # Linux/macOS
   # jasmine_env\\Scripts\\activate  # Windows
   
   # Install JASMINE
   pip install -e ".[all]"

Docker Installation
-------------------

For containerized deployment:

.. code-block:: dockerfile

   FROM python:3.9-slim
   
   # Install system dependencies
   RUN apt-update && apt-get install -y git
   
   # Clone and install JASMINE
   RUN git clone https://github.com/jaseempaloth/JASMINE.git
   WORKDIR /JASMINE
   RUN pip install -e ".[all]"
   
   # Set entrypoint
   CMD ["python"]

.. code-block:: bash

   # Build and run
   docker build -t jasmine .
   docker run -it jasmine

Updating JASMINE
-----------------

To update to the latest version:

.. code-block:: bash

   cd JASMINE
   git pull origin main
   pip install -e ".[all]" --upgrade

Uninstalling
------------

.. code-block:: bash

   # Uninstall JASMINE
   pip uninstall jasmine
   
   # Optionally remove dependencies
   pip uninstall jax jaxlib numpy

Next Steps
----------

After installation:

1. Read the :doc:`quickstart` guide
2. Try the :doc:`examples`
3. Explore the :doc:`api/index`

For issues not covered here, please visit our GitHub repository or contact support.
