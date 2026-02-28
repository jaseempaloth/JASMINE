import jax 
import jax.numpy as jnp

def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """
    Split arrays into random train and test subsets.
    
    Args:
        X (jnp.ndarray): Input features.
        y (jnp.ndarray): Target labels.
        test_size (float): Proportion of the dataset to include in the test split (0.0 to 1.0).
        shuffle (bool): Whether to shuffle the data before splitting.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0.0 and 1.0")
    
    n_samples = X.shape[0]
    n_test = max(1, int(n_samples * test_size))  # Ensure at least 1 test sample
    n_train = n_samples - n_test

    indices = jnp.arange(n_samples)
    if shuffle:
        key = jax.random.PRNGKey(random_state) if random_state is not None else jax.random.PRNGKey(0)
        indices = jax.random.permutation(key, indices)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test
