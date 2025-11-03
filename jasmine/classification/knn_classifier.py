import jax
import jax.numpy as jnp
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
        
        # Create a per-sample prediction function with partial application
        # Note: We need to handle the metric function carefully
        def predict_fn(x):
            return self._predict_single_impl(
                x, self.X_train, self.y_train, self.n_neighbors, self.n_classes
            )
        return jax.vmap(predict_fn)(X_test)
    
    def _predict_single_impl(
        self,
        x_test_single: jnp.ndarray,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        n_neighbors: int,
        n_classes: int,
    ) -> jnp.ndarray:
        """
        Internal implementation for predicting a single sample.
        """
        # Compute distances from the test instance to all training instances.
        distances = self.metric(x_test_single, X_train)

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