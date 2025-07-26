import sys
import os
# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import jax.numpy as jnp
from jasmine.linear_model import LinearRegression
from jasmine.metrics.loss import MSELoss, MAELoss, RMSELoss


def simple_test():
    """A simple test to verify LinearRegression works with different loss functions."""
    print("=== Simple Linear Regression Test ===\n")
    
    # Create simple test data
    # X has 2 features, y = 2*x1 + 3*x2 + 1 (approximately)
    X = jnp.array([
        [1.0, 1.0],
        [2.0, 1.0], 
        [1.0, 2.0],
        [2.0, 2.0]
    ])
    y = jnp.array([6.0, 8.0, 9.0, 11.0])  # 2*1+3*1+1=6, 2*2+3*1+1=8, etc.
    
    print("Training data:")
    print(f"X = \n{X}")
    print(f"y = {y}")
    print(f"Expected relationship: y ≈ 2*x1 + 3*x2 + 1\n")
    
    # Test with different loss functions
    loss_functions = [MSELoss, MAELoss, RMSELoss]
    
    for loss_fn in loss_functions:
        print(f"--- Testing with {loss_fn.__name__} ---")
        
        # Create and train model
        model = LinearRegression(loss=loss_fn)
        model.fit(X, y, learning_rate=0.1, max_iter=1000)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Calculate final loss
        final_loss = loss_fn()(model.params_, X, y, model)
        
        # Print results
        print(f"Learned coefficients: {model.coef_}")
        print(f"Learned intercept: {model.intercept_}")
        print(f"Predictions: {predictions}")
        print(f"True values: {y}")
        print(f"Final loss: {final_loss}")
        print(f"R² score: {model.score(X, y):.4f}")
        print()


def very_simple_test():
    """An even simpler test with just 2 data points."""
    print("=== Very Simple Test (2 points) ===\n")
    
    # Super simple: y = x + 2
    X = jnp.array([[1.0], [2.0]])  # Just one feature
    y = jnp.array([3.0, 4.0])      # y = x + 2
    
    print("Training data:")
    print(f"X = {X.flatten()}")
    print(f"y = {y}")
    print("Expected: y = x + 2 (coefficient=1, intercept=2)\n")
    
    # Test with MSE loss
    model = LinearRegression(loss=MSELoss)
    model.fit(X, y, learning_rate=0.1, max_iter=500)
    
    predictions = model.predict(X)
    
    print("Results:")
    print(f"Learned coefficient: {model.coef_[0]:.4f} (expected: 1.0)")
    print(f"Learned intercept: {model.intercept_:.4f} (expected: 2.0)")
    print(f"Predictions: {predictions}")
    print(f"True values: {y}")
    print(f"Difference: {jnp.abs(predictions - y)}")


if __name__ == "__main__":
    try:
        very_simple_test()
        print("\n" + "="*50 + "\n")
        simple_test()
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
