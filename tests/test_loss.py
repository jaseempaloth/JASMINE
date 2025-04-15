import sys
import os
# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pytest
import jax
from jax import numpy as jnp
import numpy as np
from jasmine.metrics.loss import MSELoss, MAELoss, RMSELoss, CrossEntropyLoss


class MockModel:
    """A mock model class to test loss functions."""
    
    def __init__(self, return_value):
        """
        Initialize the mock model with a predetermined return value.
        
        Args:
            return_value (jnp.ndarray): The value to return from forward
        """
        self.return_value = return_value
        
    def forward(self, params, X):
        """
        Mock forward method that returns the predetermined value.
        
        Args:
            params (dict): Ignored
            X (jnp.ndarray): Ignored
            
        Returns:
            jnp.ndarray: The predetermined return value
        """
        return self.return_value


class TestMSELoss:
    
    def test_mse_simple(self):
        """Test MSE loss with simple values."""
        # Setup
        params = {}
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([1.0, 2.0])
        preds = jnp.array([2.0, 4.0])  # (2.0-1.0)^2 + (4.0-2.0)^2 / 2 = 1 + 4 / 2 = 2.5
        model = MockModel(preds)
        loss_fn = MSELoss()
        
        # Execute
        loss_value = loss_fn(params, X, y, model)
        
        # Verify
        assert jnp.isclose(loss_value, 2.5)
    
    def test_mse_zeros(self):
        """Test MSE loss with zero predictions."""
        params = {}
        X = jnp.zeros((3, 2))
        y = jnp.array([1.0, 2.0, 3.0])
        preds = jnp.zeros(3)  # (0-1)^2 + (0-2)^2 + (0-3)^2 / 3 = 1 + 4 + 9 / 3 = 14/3
        model = MockModel(preds)
        loss_fn = MSELoss()
        
        loss_value = loss_fn(params, X, y, model)
        
        expected = (1.0**2 + 2.0**2 + 3.0**2) / 3
        assert jnp.isclose(loss_value, expected)
    
    def test_mse_gradient(self):
        """Test the gradient computation of MSE loss."""
        # Setup a simple case where we can predict the gradient
        key = jax.random.PRNGKey(0)
        params = {'w': jnp.array([1.0])}
        X = jnp.array([[1.0], [2.0]])
        y = jnp.array([2.0, 3.0])
        
        # Create a model that returns X * w
        class LinearModel:
            def forward(self, params, X):
                return X @ params['w']
        
        model = LinearModel()
        loss_fn = MSELoss()
        
        # Execute
        grad_value = loss_fn.compute_grad(params, X, y, model)
        
        # For this simple case, we can calculate the expected gradient manually
        # d/dw MSE = d/dw (1/n)∑(X*w - y)² = (2/n)∑(X*w - y)*X
        # With our values: gradient = (2/2)*((1*1-2)*1 + (2*1-3)*2) = (-1)*1 + (-1)*2 = -3
        expected_grad = {'w': jnp.array([-3.0])}
        
        assert jnp.allclose(grad_value['w'], expected_grad['w'])


class TestMAELoss:
    
    def test_mae_simple(self):
        """Test MAE loss with simple values."""
        params = {}
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([1.0, 2.0])
        preds = jnp.array([2.0, 4.0])  # |2.0-1.0| + |4.0-2.0| / 2 = 1 + 2 / 2 = 1.5
        model = MockModel(preds)
        loss_fn = MAELoss()
        
        loss_value = loss_fn(params, X, y, model)
        
        assert jnp.isclose(loss_value, 1.5)
    
    def test_mae_negative_diff(self):
        """Test MAE loss with negative differences."""
        params = {}
        X = jnp.zeros((3, 2))
        y = jnp.array([1.0, -2.0, 3.0])
        preds = jnp.array([0.0, 0.0, 0.0])  # |0-1| + |0-(-2)| + |0-3| / 3 = 1 + 2 + 3 / 3 = 2
        model = MockModel(preds)
        loss_fn = MAELoss()
        
        loss_value = loss_fn(params, X, y, model)
        
        expected = (1.0 + 2.0 + 3.0) / 3
        assert jnp.isclose(loss_value, expected)


class TestRMSELoss:
    
    def test_rmse_simple(self):
        """Test RMSE loss with simple values."""
        params = {}
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([1.0, 2.0])
        preds = jnp.array([2.0, 4.0])  # sqrt((2.0-1.0)^2 + (4.0-2.0)^2 / 2) = sqrt(2.5) ≈ 1.58
        model = MockModel(preds)
        loss_fn = RMSELoss()
        
        loss_value = loss_fn(params, X, y, model)
        
        expected = jnp.sqrt((1.0**2 + 2.0**2) / 2)
        print(f"RMSE simple test - Computed loss: {loss_value}, Expected: {expected}")
        assert jnp.isclose(loss_value, expected)
    
    def test_rmse_zeros(self):
        """Test RMSE with zero predictions."""
        params = {}
        X = jnp.zeros((3, 2))
        y = jnp.array([3.0, 4.0, 5.0])
        preds = jnp.zeros(3)  # sqrt((3^2 + 4^2 + 5^2) / 3) = sqrt(50/3) ≈ 4.08
        model = MockModel(preds)
        loss_fn = RMSELoss()
        
        loss_value = loss_fn(params, X, y, model)
        
        expected = jnp.sqrt((3.0**2 + 4.0**2 + 5.0**2) / 3)
        print(f"RMSE zeros test - Computed loss: {loss_value}, Expected: {expected}")
        assert jnp.isclose(loss_value, expected)


class TestCrossEntropyLoss:
    
    def test_binary_cross_entropy(self):
        """Test binary cross entropy loss."""
        params = {}
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([0.0, 1.0])
        
        # Create logits that will result in sigmoid values close to [0.1, 0.9]
        logits = jnp.array([-2.2, 2.2])
        sigmoid_probs = jax.nn.sigmoid(logits)  # ~[0.1, 0.9]
        
        model = MockModel(logits)
        loss_fn = CrossEntropyLoss()
        
        loss_value = loss_fn(params, X, y, model)
        
        # Calculate expected loss manually
        expected = -jnp.mean(y * jnp.log(sigmoid_probs) + 
                           (1 - y) * jnp.log(1 - sigmoid_probs))
        assert jnp.isclose(loss_value, expected)
    
    def test_multiclass_cross_entropy(self):
        """Test multi-class cross entropy loss."""
        params = {}
        X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        # One-hot encoded labels for 3 classes
        y = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Create logits that will give predictable softmax probabilities
        logits = jnp.array([
            [5.0, 1.0, 1.0],  # Class 0 has highest logit
            [1.0, 5.0, 1.0],  # Class 1 has highest logit
            [1.0, 1.0, 5.0]   # Class 2 has highest logit
        ])
        
        softmax_probs = jax.nn.softmax(logits, axis=1)
        
        model = MockModel(logits)
        loss_fn = CrossEntropyLoss()
        
        loss_value = loss_fn(params, X, y, model)
        
        # Calculate expected loss manually
        expected = -jnp.mean(jnp.sum(y * jnp.log(softmax_probs), axis=1))
        assert jnp.isclose(loss_value, expected)


if __name__ == "__main__":
    # Run tests manually
    test_instance = TestMSELoss()
    test_instance.test_mse_simple()
    test_instance.test_mse_zeros()
    test_instance.test_mse_gradient()
    
    test_instance = TestMAELoss()
    test_instance.test_mae_simple()
    test_instance.test_mae_negative_diff()
    
    test_instance = TestRMSELoss()
    test_instance.test_rmse_simple()
    test_instance.test_rmse_zeros()
    
    test_instance = TestCrossEntropyLoss()
    test_instance.test_binary_cross_entropy()
    test_instance.test_multiclass_cross_entropy()
    
    print("All tests passed!")