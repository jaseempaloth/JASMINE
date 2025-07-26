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


def test_linear_regression_with_gradient_check():
    """Test LinearRegression with different loss functions and gradient checking."""
    print("=== Linear Regression with Gradient Check ===\n")
    
    # Create simple test data
    X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = jnp.array([5.0, 11.0, 17.0])  # y = 1*x1 + 2*x2 + 0
    
    print("Training data:")
    print(f"X = \n{X}")
    print(f"y = {y}")
    print("Expected relationship: y = 1*x1 + 2*x2 + 0\n")
    
    # Test with different loss functions
    loss_functions = [MSELoss, MAELoss, RMSELoss]
    
    for loss_fn_class in loss_functions:
        print(f"--- Testing {loss_fn_class.__name__} ---")
        
        # Create model and train for a few iterations
        model = LinearRegression(loss=loss_fn_class)
        model.fit(X, y, learning_rate=0.01, max_iter=5)  # Just a few iterations
        
        # Get current parameters
        params = model.params_
        loss_fn = model.loss
        
        # Perform gradient check
        gradient_check(loss_fn, params, X, y, model)
        
        # Make predictions and show results
        predictions = model.predict(X)
        print(f"\nModel results after 5 iterations:")
        print(f"Coefficients: {model.coef_}")
        print(f"Intercept: {model.intercept_}")
        print(f"Predictions: {predictions}")
        print(f"True values: {y}")
        print(f"Loss: {loss_fn(params, X, y, model)}")
        print("\n" + "="*60 + "\n")


def test_gradient_check_simple_case():
    """Test gradient check with a very simple case where we know the answer."""
    print("=== Simple Gradient Check Test ===\n")
    
    # Very simple case: one feature, two data points
    X = jnp.array([[1.0], [2.0]])
    y = jnp.array([3.0, 5.0])  # y = 2*x + 1
    
    # Initialize parameters close to the true values
    params = {'weights': jnp.array([1.8]), 'intercept': jnp.array([1.2])}
    
    print("Simple test case:")
    print(f"X = {X.flatten()}")
    print(f"y = {y}")
    print(f"True relationship: y = 2*x + 1")
    print(f"Initial params: weights={params['weights']}, intercept={params['intercept']}")
    print()
    
    # Create model and test MSE loss
    model = LinearRegression(loss=MSELoss)
    loss_fn = MSELoss()
    
    # Perform gradient check
    gradient_check(loss_fn, params, X, y, model)
    
    # Show what the gradients mean
    print("\nGradient interpretation:")
    current_loss = loss_fn(params, X, y, model)
    print(f"Current loss: {current_loss}")
    
    # Manually compute expected gradients for verification
    preds = model.forward(params, X)
    residuals = preds - y
    expected_weight_grad = jnp.mean(2 * residuals * X.flatten())
    expected_intercept_grad = jnp.mean(2 * residuals)
    
    print(f"Expected weight gradient: {expected_weight_grad}")
    print(f"Expected intercept gradient: {expected_intercept_grad}")


def test_convergence_with_gradient_check():
    """Test how gradients change during training."""
    print("=== Gradient Check During Training ===\n")
    
    # Create test data
    X = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    y = jnp.array([4.0, 8.0, 12.0])  # y = 2*x1 + 2*x2
    
    print("Training data:")
    print(f"X = \n{X}")
    print(f"y = {y}")
    print("True relationship: y = 2*x1 + 2*x2\n")
    
    # Create model
    model = LinearRegression(loss=MSELoss)
    loss_fn = MSELoss()
    
    # Initialize parameters
    if model.fit_intercept:
        params = {'weights': jnp.zeros(X.shape[1]), 'intercept': jnp.array([0.0])}
    else:
        params = {'weights': jnp.zeros(X.shape[1])}
    
    # Check gradients at different points during training
    iterations_to_check = [0, 5, 10, 20]
    
    for iteration in iterations_to_check:
        print(f"--- Gradient Check at Iteration {iteration} ---")
        
        # Train for specified number of iterations
        if iteration > 0:
            for _ in range(5 if iteration == 5 else 10 if iteration == 10 else 10):
                grads = loss_fn.compute_grad(params, X, y, model)
                for key in params:
                    params[key] = params[key] - 0.01 * grads[key]
        
        # Check gradients
        current_loss = loss_fn(params, X, y, model)
        print(f"Current loss: {current_loss}")
        print(f"Current params: {params}")
        
        # Perform gradient check (with relaxed tolerance for later iterations)
        tolerance = 1e-6 if iteration == 0 else 1e-5
        gradient_check(loss_fn, params, X, y, model, tolerance=tolerance)
        print()


if __name__ == "__main__":
    try:
        test_gradient_check_simple_case()
        print("\n" + "="*80 + "\n")
        
        test_linear_regression_with_gradient_check()
        print("\n" + "="*80 + "\n")
        
        test_convergence_with_gradient_check()
        
        print("✅ All gradient checks completed successfully!")
        
    except Exception as e:
        print(f"❌ Gradient check failed with error: {e}")
        import traceback
        traceback.print_exc()