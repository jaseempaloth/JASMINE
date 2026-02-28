import jax.numpy as jnp
from jasmine.datasets import generate_classification
from jasmine.preprocessing import StandardScaler
from jasmine.model_selection import train_test_split
from jasmine.neighbors import KNNClassifier

def knn_classifier():
    # Generate synthetic classification data
    X, y = generate_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Scaled training data: {X_train_scaled[:5]}")
    print(f"Scaled test data: {X_test_scaled[:5]}")

    # Initialize the KNN classifier
    knn = KNNClassifier()
    knn.train(X_train_scaled, y_train)
    print(f"Trained KNN classifier with {knn.n_neighbors} neighbors.")

    # Perform inference on the test set
    y_pred = knn.inference(X_test_scaled)
    print(f"Predicted labels: {y_pred[:5]}")
    print(f"True labels: {y_test[:5]}")

    # Evaluate the model
    accuracy = knn.evaluate(X_test_scaled, y_test)
    print(f"Model accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    knn_classifier()
