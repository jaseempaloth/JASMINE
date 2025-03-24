import jax.numpy as jnp
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jasmine.linear_model import LogisticRegression


def test_binary_classification():
    # Binary classification example
    X = jnp.array([[0.1, 0.5], [1.2, 1.5], [2.4, 4.5], [3.5, 2.6]])
    y = jnp.array([0, 1, 1, 0])

    # Train model
    lg = LogisticRegression()
    lg.fit(X, y, learning_rate=0.01, max_iter=1000)

    # Test predictions
    preds = lg.predict(X)
    assert (preds == y).all(), f"Binary classification failed: {preds} != {y}"
    print("Binary classification test passed!")


def test_multiclass_classification():
    # Multi-class classification dataset
    X = jnp.array([[0.1, 0.5], [1.1, 1.5], [2.1, 2.5], [3.1, 3.5], [4.1, 4.5]])
    y = jnp.array([0, 1, 2, 1, 0])
    y_one_hot = jnp.eye(3)[y]
    
    # Train model
    lg = LogisticRegression()
    lg.fit(X, y_one_hot, learning_rate=0.1, max_iter=1000)
    
    # Test predictions
    preds = lg.predict(X)
    assert (preds == y).all(), f"Multi-class classification failed: {preds} != {y}"
    print("Multi-class classification test passed!")

if __name__ == "__main__":
    test_binary_classification()
    test_multiclass_classification()


