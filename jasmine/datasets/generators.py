import jax
import jax.numpy as jnp
import os
from typing import Optional, Tuple

def generate_regression(n_samples=100, n_features=20, n_informative=10, noise=0.0,
                    bias=0.0, shuffle=True, coef=False, random_state=None):
    """
    Generate a random regression problem with JAX.

    This function creates a dataset where the output is a linear combination of
    a subset of the input features, with optional Gaussian noise.

    Args:
        n_samples (int): The number of samples to generate.
        n_features (int): The total number of features.
        n_informative (int): The number of features that are actually used to
                             generate the output. The rest are noise.
        noise (float): The standard deviation of the Gaussian noise added to the output.
        bias (float): The bias term (intercept) in the underlying linear model.
        shuffle (bool): Whether to shuffle the features and informative indices.
                        If False, the informative features will always be the first
                        `n_informative` columns.
        coef (bool): If True, the ground truth coefficients and bias are returned.
        random_state (int, optional): Seed for the random number generator for
                                      reproducibility. If None, a random seed is used.

    Returns:
        tuple: By default, returns (X, y).
               If `coef` is True, returns (X, y, ground_truth_coefficients).
    """
    if n_informative > n_features:
        raise ValueError(f"n_informative ({n_informative}) cannot be greater than n_features ({n_features})")
    
    # If no seed is provided, use a secure source of randomness
    if random_state is None:
        random_state = int.from_bytes(os.urandom(4), 'big') 

    key = jax.random.PRNGKey(random_state)
    x_key, w_key, noise_key, shuffle_key = jax.random.split(key, 4)

    X = jax.random.normal(x_key, (n_samples, n_features))

    ground_truth = jnp.zeros(n_features)
    informative_weights = 100 * jax.random.normal(w_key, (n_informative,)) 

    if shuffle:
        indices = jax.random.permutation(shuffle_key, jnp.arange(n_features))
        informative_indices = indices[:n_informative]
        ground_truth = ground_truth.at[informative_indices].set(informative_weights)
    else:
        ground_truth = ground_truth.at[:n_informative].set(informative_weights)

    y = X @ ground_truth + bias

    if noise > 0:
        y += jax.random.normal(noise_key, (n_samples,)) * noise

    if coef:
        return X, y, ground_truth, bias
    else:
        return X, y
    
def generate_polynomial(n_samples: int = 100,
                        degree: int = 2,
                        noise: float = 0.0,
                        bias: float = 0.0,
                        coef: bool = False,
                        random_state: Optional[int] = None):
    """
    Generate a polynomial regression problem with one feature.

    Args:
        n_samples: The number of samples.
        degree: The degree of the polynomial relationship.
        noise: The standard deviation of the Gaussian noise.
        bias: The bias term (intercept).
        coef: If True, the ground truth coefficients and bias are returned.
        random_state: Seed for the random number generator.

    Returns:
        By default, returns (X, y). X will have a shape of (n_samples, 1).
        If `coef` is True, returns (X, y, ground_truth_coefficients, bias).
    """
    if random_state is None:
        random_state = int.from_bytes(os.urandom(4), 'big')

    key = jax.random.PRNGKey(random_state)
    x_key, w_key, noise_key = jax.random.split(key, 3)

    # Generate a single feature, sorted for easy plotting
    X = jax.random.uniform(x_key, (n_samples, 1), minval=-5, maxval=5)
    X = jnp.sort(X, axis=0)

    # Generate true coefficients for the polynomial terms (x, x^2, ..., x^degree)
    true_coefficients = jax.random.normal(w_key, (degree, )) * 5 

    # Create the polynomial features from the original feature X
    powers = jnp.arange(1, degree + 1)
    X_poly_features = X ** powers 

    # Calculate y using the polynomial equation: y = (w1*x + w2*x^2 + ...) + bias
    y = X_poly_features @ true_coefficients + bias

    if noise > 0.0:
        y += jax.random.normal(noise_key, (n_samples,)) * noise

    if coef:
        return X, y, true_coefficients, bias
    else:
        return X, y

def generate_classification(n_samples: int = 100,
                            n_features: int = 20,
                            n_informative: int = 5,
                            n_redundant: int = 2,
                            n_classes: int = 2,
                            class_sep: float = 1.0,
                            feature_noise: float = 1.0,
                            redundant_noise: float = 0.0,
                            shuffle: bool = True,
                            random_state: Optional[int] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate a random n-class classification problem with.

    This function creates clusters of points normally distributed around vertices
    of a hypercube, making it suitable for testing classification algorithms.

    Args:
        n_samples: The number of samples.
        n_features: The total number of features.
        n_informative: The number of informative features.
        n_redundant: The number of redundant features (linear combinations of informative features).
        n_classes: The number of classes (or labels).
        class_sep: Factor multiplying the hypercube size. Larger values spread
                   out the classes and make the problem easier.
        shuffle: Whether to shuffle the features.
        random_state: Seed for the random number generator.

    Returns:
        A tuple (X, y) where X is the feature matrix and y are the integer labels.
    """
    # Validate input parameters
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer, got {n_samples}")
    if n_features <= 0:
        raise ValueError("n_features must be a positive integer, got {n_features}")
    if n_informative < 0:
        raise ValueError(f"n_informative must be non-negative, got {n_informative}")
    if n_redundant < 0:
        raise ValueError(f"n_redundant must be non-negative, got {n_redundant}")
    if n_classes < 1:
        raise ValueError(f"n_classes must be at least 1, got {n_classes}")
    if class_sep <= 0:
        raise ValueError(f"class_sep must be positive, got {class_sep}")
    if feature_noise < 0:
        raise ValueError(f"feature_noise must be non-negative, got {feature_noise}")
    if redundant_noise < 0:
        raise ValueError(f"redundant_noise must be non-negative, got {redundant_noise}")
    if n_informative + n_redundant > n_features:
        raise ValueError(f"n_informative ({n_informative}) + n_redundant ({n_redundant}) cannot be greater than n_features ({n_features})")
    
    if random_state is None:
        random_state = int.from_bytes(os.urandom(4), 'big')
    
    key = jax.random.PRNGKey(random_state)
    key, centroid_key, x_key, redundant_key, shuffle_key = jax.random.split(key, 5)

    # Define centroids for each class at the vertices of a hypercube
    centroids = jax.random.choice(centroid_key, jnp.array([-class_sep, class_sep]),
                                  shape=(n_classes, n_features))

    # Assign samples to classes
    y = jax.random.randint(key, (n_samples,), 0, n_classes)

    # Generate features matrix
    X = jnp.zeros((n_samples, n_features))

    # Create informative features by adding noise to class centroids
    informative_centroids = centroids[y][:, :n_informative]
    X = X.at[:, :n_informative].set(informative_centroids + jax.random.normal(x_key, (n_samples, n_informative)))

    # Create redundant features
    if n_redundant > 0:
        w_redundant = jax.random.normal(redundant_key, (n_redundant, n_informative))
        redundant_features = X[:, :n_informative] @ w_redundant.T
        X = X.at[:, n_informative:n_informative + n_redundant].set(redundant_features)
    
    # Fill remaining features with noise
    n_noise = n_features - n_informative - n_redundant
    if n_noise > 0:
        noise_features = jax.random.normal(key, (n_samples, n_noise))
        X = X.at[:, -n_noise:].set(noise_features)
    
    if shuffle:
        X = jax.random.permutation(shuffle_key, X)
    
    return X, y
        

