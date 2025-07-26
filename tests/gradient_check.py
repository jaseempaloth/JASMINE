import sys
import os
# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import jax
import jax.numpy as jnp
from jasmine.linear_model import LinearRegression
from jasmine.metrics.loss import MSELoss, MAELoss, RMSELoss


def numerical_gradient(loss_fn, params, X, y, model, epsilon=1e-5):
    """
    Compute numerical gradients using finite differences.
    
    Args:
        loss_fn: Loss function
        params (dict): Model parameters
        X (jnp.ndarray): Input features
        y (jnp.ndarray): Target values
        model: Model instance
        epsilon (float): Small value for finite difference
        
    Returns:
        dict: Numerical gradients
    """
    numerical_grads = {}
    
    for param_name, param_value in params.items():
        numerical_grads[param_name] = jnp.zeros_like(param_value)
        
        # For each element in the parameter
        flat_param = param_value.flatten()
        flat_grad = jnp.zeros_like(flat_param)
        
        for i in range(len(flat_param)):
            # Create modified parameters for forward difference
            params_plus = params.copy()
            params_minus = params.copy()
            
            # Modify the i-th element
            flat_param_plus = flat_param.at[i].add(epsilon)
            flat_param_minus = flat_param.at[i].add(-epsilon)
            
            # Reshape back to original shape
            params_plus[param_name] = flat_param_plus.reshape(param_value.shape)
            params_minus[param_name] = flat_param_minus.reshape(param_value.shape)
            
            # Compute loss with modified parameters
            loss_plus = loss_fn(params_plus, X, y, model)
            loss_minus = loss_fn(params_minus, X, y, model)
            
            # Compute numerical gradient
            flat_grad = flat_grad.at[i].set((loss_plus - loss_minus) / (2 * epsilon))
        
        numerical_grads[param_name] = flat_grad.reshape(param_value.shape)
    
    return numerical_grads


def gradient_check(loss_fn, params, X, y, model, epsilon=1e-5, tolerance=1e-6):
    """
    Check if analytical gradients match numerical gradients.
    
    Args:
        loss_fn: Loss function instance
        params (dict): Model parameters
        X (jnp.ndarray): Input features
        y (jnp.ndarray): Target values
        model: Model instance
        epsilon (float): Small value for finite difference
        tolerance (float): Tolerance for gradient comparison
        
    Returns:
        bool: True if gradients match, False otherwise
    """
    print("=== Gradient Check ===")
    
    # Compute analytical gradients
    analytical_grads = loss_fn.compute_grad(params, X, y, model)
    
    # Compute numerical gradients
    numerical_grads = numerical_gradient(loss_fn, params, X, y, model, epsilon)
    
    # Compare gradients
    all_close = True
    
    for param_name in params.keys():
        analytical = analytical_grads[param_name]
        numerical = numerical_grads[param_name]
        
        # Compute relative error
        diff = jnp.abs(analytical - numerical)
        relative_error = diff / (jnp.abs(analytical) + jnp.abs(numerical) + 1e-8)
        max_relative_error = jnp.max(relative_error)
        
        print(f"\nParameter: {param_name}")
        print(f"Analytical gradient: {analytical}")
        print(f"Numerical gradient:  {numerical}")
        print(f"Absolute difference: {diff}")
        print(f"Max relative error:  {max_relative_error:.2e}")
        
        # Check if gradients are close
        is_close = jnp.allclose(analytical, numerical, atol=tolerance, rtol=tolerance)
        print(f"Gradients match: {is_close}")
        
        if not is_close:
            all_close = False
    
    print(f"\nOverall gradient check: {'PASSED' if all_close else 'FAILED'}")
    return all_close


def test_gradient_check_simple():
    """Test gradient check with a simple linear regression example."""
    print("=== Testing Gradient Check with Simple Linear Regression ===\n")
    
    # Create simple test data
    X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([5.0, 11.0])  # y = 1*x1 + 2*x2 + 0 (perfect linear relationship)
    
    # Initialize parameters manually
    params = {'weights': jnp.array([0.5, 1.5]), 'intercept': jnp.array([0.2])}
    
    # Create model and loss function
    model = LinearRegression(loss=MSELoss)
    loss_fn = MSELoss()
    
    print("Test data:")
    print(f"X = \n{X}")
    print(f"y = {y}")
    print(f"Initial params: {params}")
    print()
    
    # Test gradient check for different loss functions
    loss_functions = [MSELoss, MAELoss, RMSELoss]
    
    for loss_class in loss_functions:
        print(f"--- Testing {loss_class.__name__} ---")
        loss_fn = loss_class()
        model.loss = loss_fn
        
        # Perform gradient check
        gradient_check(loss_fn, params, X, y, model)
        print("\n" + "="*50 + "\n")


def test_gradient_check_trained_model():
    """Test gradient check with a trained model."""
    print("=== Testing Gradient Check with Trained Model ===\n")
    
    # Create test data
    X = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y = jnp.array([3.0, 6.0, 9.0])
    
    # Train a model
    model = LinearRegression(loss=MSELoss)
    model.fit(X, y, learning_rate=0.1, max_iter=100)
    
    print("Trained model parameters:")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print()
    
    # Use the trained parameters for gradient check
    params = model.params_
    loss_fn = model.loss
    
    # Perform gradient check
    gradient_check(loss_fn, params, X, y, model)


if __name__ == "__main__":
    try:
        test_gradient_check_simple()
        test_gradient_check_trained_model()
        print("✅ All gradient checks completed!")
        
    except Exception as e:
        print(f"❌ Gradient check failed with error: {e}")
        import traceback
        traceback.print_exc()
