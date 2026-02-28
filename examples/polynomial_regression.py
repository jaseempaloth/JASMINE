import jax.numpy as jnp
import matplotlib.pyplot as plt
from jasmine.linear_model import LinearRegression
from jasmine.datasets import generate_polynomial
from jasmine.model_selection import train_test_split

def main():
    # Define parameters for data generation - REDUCED for stability
    degree = 3
    n_samples = 500 
    noise = 0.2 

    # Generate synthetic polynomial regression data
    # X_orig has one column, representing the original feature 'x'
    X_orig, y, true_coeffs, true_bias = generate_polynomial(
        n_samples=n_samples,
        degree=degree,
        noise=noise,
        bias=1.0,     # Reduced from 2.0
        coef=True,
        random_state=42
    )
    print(f"Generated data with a true polynomial relationship of degree {degree}.")
    print(f"True coefficients: {true_coeffs}, True bias: {true_bias}")

    # Create polynomial features from the original data
    if X_orig.shape[1] != 1:
        raise ValueError("Input X_orig must have exactly one feature to create polynomial features.")
    powers = jnp.arange(1, degree + 1)
    X_poly = X_orig ** powers
    print(f"Transformed data from shape {X_orig.shape} to {X_poly.shape}")

    # NORMALIZE features to prevent exploding gradients
    X_poly_mean = jnp.mean(X_poly, axis=0)
    X_poly_std = jnp.std(X_poly, axis=0)
    X_poly_normalized = (X_poly - X_poly_mean) / (X_poly_std + 1e-8)
    
    print(f"Feature ranges after normalization:")
    print(f"  Min: {jnp.min(X_poly_normalized, axis=0)}")
    print(f"  Max: {jnp.max(X_poly_normalized, axis=0)}")

    # Split the *normalized* data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_poly_normalized, y, test_size=0.2, random_state=42)
    print(f"Split data into {X_train.shape[0]} training and {X_test.shape[0]} testing samples.")

    # Initialize and train with optimized settings
    model = LinearRegression(learning_rate=0.001, n_epochs=500, l2_penalty=0.01)
    
    print("\nTraining polynomial regression model...")
    history = model.train(X_train, y_train, verbose=1)
    print("Model trained successfully.")

    test_predictions = model.inference(X_test)
    print(f"Test predictions range: [{jnp.min(test_predictions):.2f}, {jnp.max(test_predictions):.2f}]")


    # Evaluate the model on the test set
    r2 = model.evaluate(X_test, y_test)
    print(f"R²: {r2:.4f}")

    # Visualize the results - SIMPLIFIED AND ROBUST
    
    # Create a smooth curve for visualization across the full range
    x_plot = jnp.linspace(jnp.min(X_orig), jnp.max(X_orig), 300).reshape(-1, 1)
    
    # Transform x_plot to polynomial features and normalize using the same stats
    x_plot_poly = x_plot ** powers
    x_plot_normalized = (x_plot_poly - X_poly_mean) / (X_poly_std + 1e-8)
    
    # Get predictions for the smooth curve
    y_plot_pred = model.inference(x_plot_normalized)
    
    plt.figure(figsize=(12, 8))
    
    # Plot all original data points
    plt.scatter(X_orig.ravel(), y, alpha=0.6, s=40, color='lightblue', 
               label=f"Original Data (n={n_samples})", zorder=1, edgecolors='navy', linewidth=0.5)
    
    # Plot the smooth polynomial fit curve
    plt.plot(x_plot.ravel(), y_plot_pred, color='red', linewidth=3, 
             label=f"Polynomial Fit (Degree {degree})", zorder=3)
    
    plt.title(f"Polynomial Regression: Degree {degree} Fit", fontsize=16, fontweight='bold')
    plt.xlabel("Original Feature (x)", fontsize=14)
    plt.ylabel("Target (y)", fontsize=14)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Add performance info to plot
    info_text = f'R² = {r2:.4f}\nSamples = {n_samples}\nNoise = {noise}\nCoeffs = {true_coeffs}'
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Set axis limits for better view
    x_margin = (jnp.max(X_orig) - jnp.min(X_orig)) * 0.05
    plt.xlim(jnp.min(X_orig) - x_margin, jnp.max(X_orig) + x_margin)
    
    plt.tight_layout()
    plt.show()
    
    print("\nVisualization completed!")
    print(f"The red curve shows the learned polynomial relationship")
    print(f"R² = {r2:.4f} indicates {r2*100:.1f}% of variance explained")

if __name__ == "__main__":
    main()
