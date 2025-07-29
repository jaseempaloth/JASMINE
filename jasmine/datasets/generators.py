import jax
import jax.numpy as jnp
import os

def make_regression(n_samples=100, n_features=20, n_informative=10, noise=0.0,
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