from jasmine import train_test_split

def test_train_test_split():
    import jax.numpy as jnp

    # Sample data
    X = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = jnp.array([0, 1, 0, 1])

    # Test with default parameters
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 1
    assert y_train.shape[0] == 3
    assert y_test.shape[0] == 1

    # Test with specific test_size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 1

    # Test with shuffle=False
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    assert (X_train == jnp.array([[1, 2], [3, 4], [5, 6]])).all()
    assert (X_test == jnp.array([[7, 8]])).all()

