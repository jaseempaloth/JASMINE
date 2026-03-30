import jax.numpy as jnp

from jasmine.datasets import generate_regression
from jasmine.linear_model import Ridge
from jasmine.metrics import mean_squared_error
from jasmine.model_selection import train_test_split


def main():
    X, y = generate_regression(
        n_samples=1000,
        n_features=20,
        n_informative=5,
        noise=0.2,
        bias=2.0,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Ridge(alpha=0.1, learning_rate=0.01, n_epochs=1000)
    model.train(X_train, y_train)

    predictions = model.inference(X_test)
    mse = model.evaluate(X_test, y_test, metrics_fn=mean_squared_error)
    r2 = model.evaluate(X_test, y_test)

    print(f"Generated data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target range: [{jnp.min(y):.2f}, {jnp.max(y):.2f}]")
    print("Model trained successfully.")
    print("Learned parameters:", model.params["w"])
    print("Learned bias:", model.params["b"])
    print(f"Predictions shape: {predictions.shape}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")


if __name__ == "__main__":
    main()
