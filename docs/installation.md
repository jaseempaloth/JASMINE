# Installation Guide

Complete installation instructions for JASMINE across different platforms and environments.

## Quick Installation

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/jaseempaloth/JASMINE.git
cd JASMINE

# Install dependencies
pip install -r requirements.txt

# Install JASMINE in development mode
pip install -e .
```

### Verify Installation

```python
import jasmine
print(f"JASMINE version: {jasmine.__version__}")

# Test basic functionality
from jasmine.regression import LinearRegression
from jasmine.datasets import generate_regression

X, y = generate_regression(n_samples=100, n_features=5, random_state=42)
model = LinearRegression()
model.train(X, y)
print("JASMINE installed successfully!")
```

## Detailed Installation

### Prerequisites

#### Python Version
- **Required**: Python 3.8 or higher
- **Recommended**: Python 3.9 or 3.10
- **Not supported**: Python 3.7 or below

Check your Python version:
```bash
python --version
# or
python3 --version
```

#### System Requirements

**Minimum:**
- 4GB RAM
- 2GB free disk space
- Intel/AMD x64 processor

**Recommended:**
- 8GB+ RAM
- 5GB+ free disk space
- Modern CPU with AVX support
- NVIDIA GPU (optional, for acceleration)

### Platform-Specific Instructions

#### macOS

##### Using Homebrew (Recommended)
```bash
# Install Python if needed
brew install python@3.10

# Clone and install JASMINE
git clone https://github.com/jaseempaloth/JASMINE.git
cd JASMINE
pip3 install -r requirements.txt
pip3 install -e .
```

##### Using Conda
```bash
# Create conda environment
conda create -n jasmine python=3.10
conda activate jasmine

# Install JASMINE
git clone https://github.com/jaseempaloth/JASMINE.git
cd JASMINE
pip install -r requirements.txt
pip install -e .
```

#### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install Python and pip if needed
sudo apt install python3 python3-pip python3-venv git

# Create virtual environment (recommended)
python3 -m venv jasmine-env
source jasmine-env/bin/activate

# Install JASMINE
git clone https://github.com/jaseempaloth/JASMINE.git
cd JASMINE
pip install -r requirements.txt
pip install -e .
```

#### Windows

##### Using Command Prompt
```cmd
# Clone repository
git clone https://github.com/jaseempaloth/JASMINE.git
cd JASMINE

# Create virtual environment
python -m venv jasmine-env
jasmine-env\Scripts\activate

# Install JASMINE
pip install -r requirements.txt
pip install -e .
```

##### Using Anaconda
```cmd
# Create conda environment
conda create -n jasmine python=3.10
conda activate jasmine

# Install JASMINE
git clone https://github.com/jaseempaloth/JASMINE.git
cd JASMINE
pip install -r requirements.txt
pip install -e .
```

### GPU Support

#### NVIDIA GPU (CUDA)

For GPU acceleration, install JAX with CUDA support:

```bash
# First, install JASMINE normally
pip install -r requirements.txt
pip install -e .

# Then upgrade to CUDA version of JAX
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Requirements:**
- NVIDIA GPU with CUDA Compute Capability 3.5+
- CUDA 11.1 or newer
- cuDNN 8.0.5 or newer

**Verify GPU support:**
```python
import jax
print("Available devices:", jax.devices())
print("Default device:", jax.devices()[0])

# Should show something like:
# Available devices: [GpuDevice(id=0, process_index=0)]
```

#### Apple Silicon (M1/M2 Macs)

JAX has experimental support for Apple Silicon:

```bash
# Install JASMINE
pip install -r requirements.txt
pip install -e .

# No additional steps needed - JAX will use Metal backend automatically
```

### Virtual Environments

#### Using venv (Recommended)

```bash
# Create virtual environment
python -m venv jasmine-env

# Activate (Linux/macOS)
source jasmine-env/bin/activate

# Activate (Windows)
jasmine-env\Scripts\activate

# Install JASMINE
pip install -r requirements.txt
pip install -e .

# Deactivate when done
deactivate
```

#### Using Conda

```bash
# Create conda environment with specific Python version
conda create -n jasmine python=3.10

# Activate environment
conda activate jasmine

# Install JASMINE
git clone https://github.com/jaseempaloth/JASMINE.git
cd JASMINE
pip install -r requirements.txt
pip install -e .

# Deactivate when done
conda deactivate
```

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **JAX** | ≥ 0.4.0 | Core computation engine (includes numerical operations) |

### Optional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **NumPy** | ≥ 1.21.0 | Interoperability with NumPy arrays |
| **SciPy** | ≥ 1.7.0 | Additional scientific functions |
| **Matplotlib** | ≥ 3.3.0 | Plotting and visualization |
| **Scikit-learn** | ≥ 1.0.0 | Comparison benchmarks |
| **Jupyter** | Latest | Interactive notebooks |

### Development Dependencies

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Includes:
# - pytest (testing)
# - black (code formatting)
# - flake8 (linting)
# - sphinx (documentation)
```

## Troubleshooting

### Common Issues

#### 1. JAX Installation Fails

**Problem**: `pip install jax` fails with compilation errors

**Solution**: Use pre-built wheels
```bash
pip install --upgrade pip
pip install jax jaxlib -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

#### 2. Import Errors

**Problem**: `ImportError: No module named 'jasmine'`

**Solution**: Ensure you installed in development mode
```bash
cd JASMINE
pip install -e .
```

#### 3. GPU Not Detected

**Problem**: `jax.devices()` only shows CPU

**Solutions**:
```bash
# Check CUDA installation
nvidia-smi

# Reinstall JAX with CUDA
pip uninstall jax jaxlib
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Check environment variables
echo $CUDA_PATH
echo $LD_LIBRARY_PATH
```

#### 4. Memory Issues

**Problem**: Out of memory during installation/import

**Solution**: 
```bash
# Install with no cache
pip install --no-cache-dir -r requirements.txt

# Or use conda for better memory management
conda install jax jaxlib
pip install -e .
```

#### 5. Python Version Issues

**Problem**: Incompatible Python version

**Solution**:
```bash
# Check current version
python --version

# Install compatible version using pyenv
pyenv install 3.10.8
pyenv local 3.10.8

# Or use conda
conda install python=3.10
```

### Platform-Specific Issues

#### macOS

**Problem**: SSL certificate errors during download
```bash
# Update certificates
/Applications/Python\ 3.x/Install\ Certificates.command
```

**Problem**: Xcode command line tools missing
```bash
xcode-select --install
```

#### Linux

**Problem**: Permission denied errors
```bash
# Don't use sudo with pip, use virtual environment instead
python3 -m venv jasmine-env
source jasmine-env/bin/activate
pip install -r requirements.txt
```

**Problem**: Missing system dependencies
```bash
# Ubuntu/Debian
sudo apt install python3-dev build-essential

# CentOS/RHEL
sudo yum install python3-devel gcc gcc-c++
```

#### Windows

**Problem**: Microsoft Visual C++ errors
- Install Microsoft Visual C++ Build Tools
- Or install Visual Studio Community

**Problem**: Long path issues
```cmd
# Enable long paths in Windows
git config --system core.longpaths true
```

### Performance Issues

#### Slow Import Times

If `import jax` takes a long time:

```bash
# Disable JAX precompilation during import
export JAX_PLATFORMS=cpu
python -c "import jax"  # Should be faster

# For permanent fix, add to ~/.bashrc or ~/.zshrc
echo 'export JAX_PLATFORMS=cpu' >> ~/.bashrc
```

#### Memory Usage

For systems with limited RAM:

```python
# Configure JAX for low memory usage
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'

import jax
```

## Verification Tests

### Basic Functionality Test

```python
# Save as test_installation.py
import jax.numpy as jnp
from jasmine.regression import LinearRegression
from jasmine.classification import LogisticRegression
from jasmine.preprocessing import StandardScaler
from jasmine.datasets import generate_regression, generate_classification

def test_linear_regression():
    X, y = generate_regression(n_samples=100, n_features=5, random_state=42)
    model = LinearRegression()
    model.train(X, y)
    r2 = model.evaluate(X, y)
    assert r2 > 0.5, f"R² too low: {r2}"
    print("Linear Regression test passed")

def test_logistic_regression():
    X, y = generate_classification(n_samples=100, n_features=5, random_state=42)
    model = LogisticRegression()
    model.train(X, y)
    from jasmine.metrics import accuracy_score
    acc = model.evaluate(X, y, metrics_fn=accuracy_score)
    assert acc > 0.7, f"Accuracy too low: {acc}"
    print("Logistic Regression test passed")

def test_preprocessing():
    X, _ = generate_regression(n_samples=100, n_features=5, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    mean = jnp.mean(X_scaled, axis=0)
    std = jnp.std(X_scaled, axis=0)
    
    assert jnp.allclose(mean, 0, atol=1e-6), f"Mean not zero: {mean}"
    assert jnp.allclose(std, 1, atol=1e-6), f"Std not one: {std}"
    print("StandardScaler test passed")

if __name__ == "__main__":
    print("Running JASMINE installation tests...")
    test_linear_regression()
    test_logistic_regression()  
    test_preprocessing()
    print("All tests passed! JASMINE is ready to use.")
```

Run the test:
```bash
python test_installation.py
```

### Performance Test

```python
# Save as test_performance.py
import time
import jax.numpy as jnp
from jasmine.regression import LinearRegression

# Generate larger dataset
X = jnp.random.normal(0, 1, (5000, 20))
y = jnp.random.normal(0, 1, (5000,))

model = LinearRegression()

# Time first run (with compilation)
start = time.time()
model.train(X, y)
first_time = time.time() - start

# Time second run (compiled)
start = time.time()
model.train(X, y)
second_time = time.time() - start

print(f"First run: {first_time:.3f}s")
print(f"Second run: {second_time:.3f}s")
print(f"Speedup: {first_time/second_time:.1f}x")

if first_time/second_time > 5:
    print("JIT compilation working correctly")
else:
    print("JIT speedup lower than expected")
```

## Getting Help

If you encounter issues during installation:

1. **Check the [FAQ](faq.md)** for common solutions
2. **Search existing [GitHub Issues](https://github.com/jaseempaloth/JASMINE/issues)**
3. **Create a new issue** with:
   - Your operating system and version
   - Python version (`python --version`)
   - Complete error message
   - Steps to reproduce the issue

## Next Steps

After successful installation:

1. Read the [Quick Start Guide](quickstart.md)
2. Try the [Examples](examples.md)
3. Check out the [API Reference](api.md)
4. Run performance tests on your hardware

Enjoy using JASMINE!
