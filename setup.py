from setuptools import setup, find_packages

setup(
    name="jasmine",
    version="0.1.0",
    author="Jaseem Paloth",
    description="JASMINE (JAX Accelerated Statistical Models and Integrated Neural Engine) - A lightweight, high-performance machine learning library built on JAX with GPU/TPU acceleration support for scalable ML workflows.",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
    ],
    python_requires=">=3.8",
)
