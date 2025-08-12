import jax
import jax.numpy as jnp
from jasmine import SVMClassifier
from jasmine.datasets import generate_classification
from jasmine.selection import train_test_split
from jasmine.preprocessing import StandardScaler

def main():
    # Generate synthetic binary classification data
    X_raw, y_raw = generate_classification(
        n_samples=1000,
        n_features=10,
        n_informative=3,
        n_redundant=2,
        n_classes=2,
        class_sep=1.0,
        shuffle=True,
        random_state=42
    )

    # Transform labels to {-1, 1}
    y = jnp.where(y_raw == 0, -1, 1)

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the SVM classifier
    svm = SVMClassifier()
    svm.train(X_train, y_train)
    print("Trained SVM classifier successfully.")

    # Inference
    y_pred = svm.inference(X_test)
    print(f"Predicted labels: {y_pred[:5]}")
    print(f"True labels: {y_test[:5]}")

    # Evaluate the model
    accuracy = svm.evaluate(X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main() 

    
