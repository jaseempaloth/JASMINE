from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="jasmine",
    version="0.1.0",
    author="Jaseem Paloth",
    author_email="jaseem@jaseempaloth.com",
    description="JASMINE (JAX Accelerated Statistical Models and Integrated Neural Engine) - A lightweight, high-performance machine learning library built on JAX with GPU/TPU acceleration support for scalable ML workflows.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaseempaloth/JASMINE",
    project_urls={
        "Bug Tracker": "https://github.com/jaseempaloth/JASMINE/issues",
        "Documentation": "https://github.com/jaseempaloth/JASMINE/docs",
        "Source Code": "https://github.com/jaseempaloth/JASMINE",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "sphinx>=4.0",
        ],
        "examples": [
            "matplotlib>=3.3.0",
            "scikit-learn>=1.0.0",
            "jupyter>=1.0.0",
        ],
        "all": [
            "pytest>=6.0",
            "black>=22.0", 
            "flake8>=4.0",
            "sphinx>=4.0",
            "matplotlib>=3.3.0",
            "scikit-learn>=1.0.0",
            "jupyter>=1.0.0",
        ],
    },
    python_requires=">=3.8",
    keywords="machine learning, jax, neural networks, linear regression, logistic regression, gpu acceleration",
    include_package_data=True,
    zip_safe=False,
)
