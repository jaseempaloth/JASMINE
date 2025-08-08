import jax
import jax.numpy as jnp
import time
from typing import Optional, Callable
from jasmine.metrics import euclidean_distance, manhattan_distance, accuracy_score

class KNNClassifier:
    """
    K-Nearest Neighbors Classifier.
    
    Args:
        n_neighbors (int): Number of neighbors to use for classification.
        metric (Callable): Distance metric function to use.
        random_state (Optional[int]): Random seed for reproducibility.
    """
    def __init__(
            self, 
            n_neighbors: int = 5,
            metric: Callable = euclidean_distance,
            random_state: Optional[int] = None
        ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.random_state = random_state

        self.X_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, X: jnp.ndarray, y: jnp.ndarray):
        """
        Train the KNN classifier model by memorizing the training data.
        
        Args:
            X (jnp.ndarray): Training features of shape (n_samples, n_features).
            y (jnp.ndarray): Training labels of shape (n_samples,).
        """
        self.X_train = X
        self.y_train = y
        self.n_classes = int(jnp.max(y) + 1)
        return self
    
    def inference(self, X_test: jnp.ndarray) -> jnp.ndarray:
        """
        Perform inference on the test data.
        
        Args:
            X_test (jnp.ndarray): Test features of shape (n_samples, n_features).
        
        Returns:
            jnp.ndarray: Predicted labels for the test data.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be trained before inference.")
        
        # Create a partial function with fixed parameters
        preditc_fn = lambda x: self.predict_single(
            x, self.X_train, self.y_train, self.n_neighbors, self.n_classes, self.metric
        )

        # Vectorize the prediction logic over the entire batch of test points
        vectorized_predict = jax.vmap(preditc_fn)

        return vectorized_predict(X_test)
    
    @staticmethod
    def predict_single(x_test_single: jnp.ndarray, X_train: jnp.ndarray, y_train: jnp.ndarray, n_neighbors: int, n_classes: int, metric: Callable) -> int:
        """
        Predict the label for a single test instance.
        
        Args:
            x_test_single (jnp.ndarray): Single test instance of shape (n_features,).
            X_train (jnp.ndarray): Training features of shape (n_samples, n_features).
            y_train (jnp.ndarray): Training labels of shape (n_samples,).
            n_neighbors (int): Number of neighbors to consider.
            n_classes (int): Number of classes in the dataset.
            metric (Callable): Distance metric function.
        
        Returns:
            jnp.ndarray: Predicted label for the test instance.
        """
        # Compute distances from the test instance to all training instances
        distances = jax.vmap(lambda x_train: metric(x_test_single, x_train))(X_train)
        
        # Get indices of the nearest neighbors
        neighbor_indices = jnp.argsort(distances)[:n_neighbors]
        
        # Get the labels of the nearest neighbors
        neighbor_labels = y_train[neighbor_indices]
        
        # Vote for the majority class.
        votes = jnp.bincount(neighbor_labels, length=n_classes)
        
        # Return the voted class label
        return jnp.argmax(votes)
    
    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray, metric_fn=accuracy_score) -> float:
        """
        Evaluate the model using the specified metrics function.
        
        Args:
            X (jnp.ndarray): Input features of shape (n_samples, n_features).
            y (jnp.ndarray): True labels of shape (n_samples,).
            metric_fn (callable): Metrics function to compute the score.
        
        Returns:
            float: Computed metrics score.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be trained before evaluation.")
        
        predictions = self.inference(X)
        return metric_fn(y, predictions)