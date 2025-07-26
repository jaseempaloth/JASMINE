import jax.numpy as jnp
import jax
from jasmine.regression import LinearRegression
from jasmine import train_test_split
import numpy as np

def test_basic_linear_regression():
    """Test basic linear regression functionality."""
    print("🔧 Testing Basic Linear Regression")
    print("=" * 40)
    
    # Create simple synthetic data
    key = jax.random.PRNGKey(42)
    n_samples, n_features = 100, 2
    
    # Generate features
    X = jax.random.normal(key, (n_samples, n_features))
    
    # True relationship: y = 2*x1 + 1*x2 + 0.5 + noise
    true_weights = jnp.array([2.0, 1.0])
    true_bias = 0.5
    noise = jax.random.normal(jax.random.PRNGKey(123), (n_samples,)) * 0.05
    y = X @ true_weights + true_bias + noise
    
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"True weights: {true_weights}")
    print(f"True bias: {true_bias}")
    
    # Initialize and train model
    model = LinearRegression(use_bias=True, learning_rate=0.02, n_epochs=500)
    print("\n🚀 Training model...")
    model.train(X, y)
    
    # Make predictions
    predictions = model.inference(X)
    mse = jnp.mean((predictions - y) ** 2)
    
    print(f"\nResults:")
    print(f"Learned weights: {model.params['w']}")
    print(f"Learned bias: {model.params['b']}")
    print(f"Final MSE: {mse:.6f}")
    
    # Simple assertions
    assert model.params is not None, "Model should have parameters after training"
    assert mse < 0.1, f"MSE should be low, got {mse}"
    
    print("✅ Basic test passed!\n")

def test_no_bias_regression():
    """Test linear regression without bias term."""
    print("🔧 Testing Linear Regression Without Bias")
    print("=" * 40)
    
    # Create data with no intercept
    key = jax.random.PRNGKey(42)
    X = jax.random.normal(key, (80, 3))
    true_weights = jnp.array([1.5, -0.8, 0.3])
    y = X @ true_weights  # No bias
    
    # Train without bias
    model = LinearRegression(use_bias=False, learning_rate=0.01, n_epochs=400)
    model.train(X, y)
    
    predictions = model.inference(X)
    mse = model.evaluate(X, y, metrics_fn="mean_squared_error")
    print(f"\nResults: {mse:.6f}")
    print(f"True weights: {true_weights}")
    print(f"Learned weights: {model.params['w']}")
    print(f"Bias: {model.params['b']}")
    print(f"MSE: {mse:.6f}")
    
    assert model.params['b'] is None, "Bias should be None"
    assert mse < 0.01, "MSE should be very low for exact linear relationship"
    
    print("✅ No bias test passed!\n")

def test_train_test_split_integration():
    """Test integration with train/test split."""
    print("🌟 Testing with Train/Test Split")
    print("=" * 40)
    
    # Generate dataset
    np.random.seed(42)
    n_samples = 150
    X = np.random.randn(n_samples, 3)
    true_weights = jnp.array([1.2, -0.7, 0.4])
    true_bias = 1.0
    y = X @ true_weights + true_bias + 0.1 * np.random.randn(n_samples)
    
    X = jnp.array(X)
    y = jnp.array(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=42
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    model = LinearRegression(use_bias=True, learning_rate=0.02, n_epochs=600)
    model.train(X_train, y_train)
    
    # Evaluate
    train_pred = model.inference(X_train)
    test_pred = model.inference(X_test)
    
    train_mse = jnp.mean((train_pred - y_train) ** 2)
    test_mse = jnp.mean((test_pred - y_test) ** 2)
    
    print(f"\nPerformance:")
    print(f"Train MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Learned weights: {model.params['w']}")
    print(f"Learned bias: {model.params['b']:.3f}")
    
    assert train_mse < 0.1, "Training MSE should be low"
    assert test_mse < 0.2, "Test MSE should be reasonable"
    assert abs(test_mse - train_mse) < 0.1, "No major overfitting"
    
    print("✅ Train/test split test passed!\n")

def test_different_loss_functions():
    """Test with different loss functions."""
    print("🎯 Testing Different Loss Functions")
    print("=" * 40)
    
    # Simple dataset
    key = jax.random.PRNGKey(42)
    X = jax.random.normal(key, (60, 2))
    y = X @ jnp.array([1.0, -0.5]) + 0.2
    
    # Import loss functions
    from jasmine.metrics._regression import mean_squared_error, mean_absolute_error
    
    loss_functions = [
        ("MSE", mean_squared_error),
        ("MAE", mean_absolute_error),
    ]
    
    for loss_name, loss_fn in loss_functions:
        print(f"\n🔍 Testing {loss_name}...")
        
        model = LinearRegression(
            use_bias=True,
            learning_rate=0.01,
            n_epochs=300,
            loss_function=loss_fn
        )
        
        model.train(X, y)
        predictions = model.inference(X)
        final_loss = loss_fn(y, predictions)
        
        print(f"{loss_name} final loss: {final_loss:.6f}")
        print(f"Weights: {model.params['w']}")
        
        assert not jnp.any(jnp.isnan(model.params['w'])), f"{loss_name} weights should not be NaN"
        assert final_loss < 1.0, f"{loss_name} loss should be reasonable"
    
    print("\n✅ Loss functions test passed!\n")

def test_large_dataset():
    """Test with larger dataset."""
    print("📊 Testing with Larger Dataset")
    print("=" * 40)
    
    # Larger synthetic dataset
    key = jax.random.PRNGKey(123)
    n_samples, n_features = 500, 5
    
    X = jax.random.normal(key, (n_samples, n_features))
    true_weights = jnp.array([0.8, -0.3, 1.2, -0.9, 0.5])
    true_bias = 2.0
    noise = jax.random.normal(jax.random.PRNGKey(456), (n_samples,)) * 0.1
    y = X @ true_weights + true_bias + noise
    
    print(f"Large dataset: {n_samples} samples, {n_features} features")
    
    # Train model
    model = LinearRegression(use_bias=True, learning_rate=0.01, n_epochs=800)
    print("🚀 Training on large dataset...")
    model.train(X, y)
    
    # Evaluate
    predictions = model.inference(X)
    mse = jnp.mean((predictions - y) ** 2)
    
    # Calculate R² score
    ss_res = jnp.sum((y - predictions) ** 2)
    ss_tot = jnp.sum((y - jnp.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\nLarge dataset results:")
    print(f"MSE: {mse:.6f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Weight difference from true: {jnp.linalg.norm(model.params['w'] - true_weights):.4f}")
    
    assert r2 > 0.95, f"R² should be high for large dataset, got {r2}"
    assert mse < 0.05, f"MSE should be low, got {mse}"
    
    print("✅ Large dataset test passed!\n")

def test_error_cases():
    """Test error handling."""
    print("⚠️ Testing Error Cases")
    print("=" * 40)
    
    model = LinearRegression()
    X = jnp.array([[1, 2], [3, 4]])
    
    # Test inference before training
    try:
        model.inference(X)
        assert False, "Should raise error"
    except ValueError as e:
        print(f"✅ Correctly caught: {str(e)[:50]}...")
    
    # Test forward with invalid parameters
    try:
        LinearRegression.forward({"w": None}, X)
        assert False, "Should raise error"
    except ValueError as e:
        print(f"✅ Correctly caught: {str(e)[:50]}...")
    
    print("✅ Error handling test passed!\n")

if __name__ == "__main__":
    print("🌸 JASMINE Linear Regression Test Suite 🌸\n")
    
    # Run all tests
    test_basic_linear_regression()
    test_no_bias_regression()
    test_train_test_split_integration()
    test_different_loss_functions()
    test_large_dataset()
    test_error_cases()
    
    print("🎉 All tests passed! Linear Regression is working perfectly! ✅")
