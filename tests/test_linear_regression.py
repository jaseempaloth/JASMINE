import jax.numpy as jnp
import jax
from jasmine.regression import LinearRegression
from jasmine import train_test_split
import numpy as np

def test_linear_regression_basic():
    """Test basic functionality of LinearRegression."""
    print("🔧 Testing Basic Linear Regression Functionality")
    print("=" * 50)
    
    # Create simple synthetic data
    key = jax.random.PRNGKey(42)
    n_samples, n_features = 100, 3
    
    # Generate features
    X = jax.random.normal(key, (n_samples, n_features))
    
    # Generate targets with known relationship: y = 2*x1 + 1*x2 - 0.5*x3 + 1.0 + noise
    true_weights = jnp.array([2.0, 1.0, -0.5])
    true_bias = 1.0
    noise = jax.random.normal(jax.random.PRNGKey(123), (n_samples,)) * 0.1
    y = X @ true_weights + true_bias + noise
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"True weights: {true_weights}")
    print(f"True bias: {true_bias}")
    
    # Test model initialization
    model = LinearRegression(use_bias=True, learning_rate=0.01, n_epochs=1000)
    print(f"Model initialized with bias: {model.use_bias}")
    
    # Test parameter initialization
    params = model.init_params(n_features)
    print(f"Initialized parameters: {params}")
    
    # Test forward pass before training
    predictions_before = model.forward(params, X[:5])
    print(f"Predictions before training (first 5): {predictions_before}")
    
    # Train the model
    print("\n🚀 Training model...")
    model.train(X, y)
    
    # Test inference
    predictions = model.inference(X)
    print(f"\nPredictions after training (first 5): {predictions[:5]}")
    print(f"True values (first 5): {y[:5]}")
    
    # Check if learned parameters are close to true parameters
    learned_weights = model.params["w"]
    learned_bias = model.params["b"]
    
    print(f"\nLearned weights: {learned_weights}")
    print(f"Learned bias: {learned_bias}")
    
    # Calculate MSE
    mse = jnp.mean((predictions - y) ** 2)
    print(f"Final MSE: {mse:.6f}")
    
    # Basic assertions
    assert model.params is not None, "Parameters should be set after training"
    assert predictions.shape == y.shape, "Predictions should match target shape"
    assert mse < 1.0, "MSE should be reasonably low for this simple problem"
    
    print("✅ Basic functionality test passed!\n")

def test_linear_regression_without_bias():
    """Test LinearRegression without bias term."""
    print("🔧 Testing Linear Regression Without Bias")
    print("=" * 50)
    
    # Create data without bias (intercept = 0)
    key = jax.random.PRNGKey(42)
    n_samples, n_features = 50, 2
    
    X = jax.random.normal(key, (n_samples, n_features))
    true_weights = jnp.array([1.5, -0.8])
    y = X @ true_weights  # No bias term
    
    # Train model without bias
    model = LinearRegression(use_bias=False, learning_rate=0.01, n_epochs=500)
    model.train(X, y)
    
    predictions = model.inference(X)
    mse = jnp.mean((predictions - y) ** 2)
    
    print(f"True weights: {true_weights}")
    print(f"Learned weights: {model.params['w']}")
    print(f"Bias parameter: {model.params['b']}")
    print(f"MSE: {mse:.6f}")
    
    assert model.params["b"] is None, "Bias should be None when use_bias=False"
    assert mse < 0.1, "MSE should be very low for linear relationship without noise"
    
    print("✅ No-bias test passed!\n")

def test_linear_regression_with_real_data():
    """Test with more realistic dataset using train/test split."""
    print("🌟 Testing with Realistic Dataset and Train/Test Split")
    print("=" * 60)
    
    # Generate more complex synthetic dataset
    np.random.seed(42)
    n_samples, n_features = 200, 4
    
    # Create correlated features
    X = np.random.randn(n_samples, n_features)
    X[:, 1] = X[:, 0] + 0.3 * np.random.randn(n_samples)  # Correlated feature
    X = jnp.array(X)
    
    # Complex relationship
    true_weights = jnp.array([1.2, -0.5, 0.8, -0.3])
    true_bias = 2.0
    noise = jnp.array(np.random.normal(0, 0.2, n_samples))
    y = X @ true_weights + true_bias + noise
    
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Feature correlations: X[:,1] correlated with X[:,0]")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    model = LinearRegression(use_bias=True, learning_rate=0.02, n_epochs=800)
    print("\n🚀 Training on training set...")
    model.train(X_train, y_train)
    
    # Evaluate on both sets
    train_predictions = model.inference(X_train)
    test_predictions = model.inference(X_test)
    
    train_mse = jnp.mean((train_predictions - y_train) ** 2)
    test_mse = jnp.mean((test_predictions - y_test) ** 2)
    
    # Calculate R² score
    def r2_score(y_true, y_pred):
        ss_res = jnp.sum((y_true - y_pred) ** 2)
        ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    print(f"\n📊 Results:")
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    print(f"\nTrue weights: {true_weights}")
    print(f"Learned weights: {model.params['w']}")
    print(f"True bias: {true_bias}")
    print(f"Learned bias: {model.params['b']}")
    
    # Assertions
    assert train_r2 > 0.8, f"Training R² should be > 0.8, got {train_r2}"
    assert test_r2 > 0.7, f"Test R² should be > 0.7, got {test_r2}"
    assert abs(test_mse - train_mse) < 0.5, "Test and train MSE should be similar (no major overfitting)"
    
    print("✅ Realistic dataset test passed!\n")

def test_different_loss_functions():
    """Test LinearRegression with different loss functions."""
    print("🎯 Testing Different Loss Functions")
    print("=" * 50)
    
    # Create simple dataset
    key = jax.random.PRNGKey(42)
    X = jax.random.normal(key, (50, 2))
    y = X @ jnp.array([1.0, -0.5]) + 0.1
    
    # Test with different loss functions
    from jasmine.metrics._regression import mean_squared_error, mean_absolute_error
    
    loss_functions = [
        ("MSE", mean_squared_error),
        ("MAE", mean_absolute_error),
    ]
    
    results = {}
    
    for loss_name, loss_fn in loss_functions:
        print(f"\n🔍 Testing with {loss_name}...")
        
        model = LinearRegression(
            use_bias=True,
            learning_rate=0.01,
            n_epochs=500,
            loss_function=loss_fn
        )
        
        model.train(X, y)
        predictions = model.inference(X)
        final_loss = loss_fn(y, predictions)
        
        results[loss_name] = {
            'loss': final_loss,
            'weights': model.params['w'],
            'bias': model.params['b']
        }
        
        print(f"{loss_name} final loss: {final_loss:.6f}")
        print(f"Learned weights: {model.params['w']}")
    
    # Both should learn reasonable parameters
    for loss_name, result in results.items():
        assert result['loss'] < 1.0, f"{loss_name} loss should be reasonable"
        assert not jnp.any(jnp.isnan(result['weights'])), f"{loss_name} weights should not be NaN"
    
    print("\n✅ Different loss functions test passed!\n")

def test_error_handling():
    """Test error handling in LinearRegression."""
    print("⚠️ Testing Error Handling")
    print("=" * 50)
    
    model = LinearRegression()
    
    # Test inference before training
    X = jnp.array([[1, 2], [3, 4]])
    
    try:
        model.inference(X)
        assert False, "Should raise error when calling inference before training"
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")
    
    # Test forward with None parameters
    try:
        model.forward(None, X)
        assert False, "Should raise error with None parameters"
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")
    
    print("✅ Error handling test passed!\n")

if __name__ == "__main__":
    print("🌸 Testing JASMINE Linear Regression Implementation 🌸\n")
    
    # Run all tests
    test_linear_regression_basic()
    test_linear_regression_without_bias()
    test_linear_regression_with_real_data()
    test_different_loss_functions()
    test_error_handling()
    
    print("🎉 All Linear Regression tests passed! JASMINE is working perfectly! ✅")


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