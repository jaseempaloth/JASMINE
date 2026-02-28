import jax.numpy as jnp
from jasmine.linear_model import LinearRegression
from jasmine.metrics import mean_squared_error
from jasmine.model_selection import train_test_split
from jasmine.datasets import generate_regression

def main():
    # Generate synthetic regression data
    X, y = generate_regression(
        n_samples=1000,
        n_features=20,
        n_informative=5,
        noise=0.1,
        bias=2.0,
        random_state=42
    )

    print(f"Generated data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target range: [{jnp.min(y):.2f}, {jnp.max(y):.2f}]")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Testing set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.train(X_train, y_train)
    print("Model trained successfully.")

    # Compare learned vs true parameters
    print("Learned parameters:", model.params["w"])
    print("Learned Bias:", model.params["b"])

    # Make predictions on the test set
    predictions = model.inference(X_test)
    print(f"Predictions shape: {predictions.shape}")


    # Evaluate the model
    mse = model.evaluate(X_test, y_test, metrics_fn=mean_squared_error)
    r2 = model.evaluate(X_test, y_test)
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")


if __name__ == "__main__":
    main()

