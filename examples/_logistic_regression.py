import jax.numpy as jnp
from jasmine.classification import LogisticRegression
from jasmine.metrics import binary_cross_entropy, accuracy_score
from jasmine.selection import train_test_split
from jasmine.datasets import generate_classification

def main():
    # Generate synthetic binary classification data
    X, y = generate_classification(
        n_samples=1000,
        n_features=20,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        class_sep=1.0,
        shuffle=True,
        random_state=42
    )
    print(f"Generated data: {X.shape[0]} samples, {X.shape[1]} features ")
    print(f"Target range: [{jnp.min(y):.2f}, {jnp.max(y):.2f}]")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Testing set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Initialize and train the logistic regression model
    model = LogisticRegression()
    model.train(X_train, y_train)
    print("Model trained successfully.")

if __name__ == "__main__":
    main()