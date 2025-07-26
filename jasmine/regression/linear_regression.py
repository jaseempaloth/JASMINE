import jax 
import jax.numpy as jnp
from ..metrics._regression import mean_squared_error, mean_absolute_error, root_mean_squared_error

class LinearRegression:
    def __init__(self, use_bias=True, learning_rate=0.01, n_epochs=1000, loss_function=mean_squared_error):
        self.use_bias = use_bias
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.loss_function = loss_function

        # Model parameters
        self.params = None

    def init_params(self, n_features, key=None):
        """
        Initialize model parameters.
        
        Args:
            n_features (int): Number of features in the input data.
            key (jax.random.PRNGKey, optional): Random key for parameter initialization.
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        w_key, _ = jax.random.split(key)

        params = {
            "w" : jax.random.normal(w_key, (n_features,))
        }

        if self.use_bias:
            params["b"] = jnp.array(0.0)
        else:
            params["b"] = None

        self.params = params
        return params

    @staticmethod
    def forward(params, X):
        """
        Forward pass for the linear model.
        
        Args:
            params (dict): Model parameters
            X (jnp.ndarray): Input features
            
        Returns:
            jnp.ndarray: Predicted values
        """
        use_bias = "b" in params and params["b"] is not None

        if params["w"] is None:
            raise ValueError("Model weights must be initialized before calling forward.")
        
        if use_bias:
            return X @ params["w"] + params["b"]
        else:
            return X @ params["w"]
        
    def loss_fn(self, params, X, y):
        """
        Compute the loss for the linear model.
        
        Args:
            params (dict): Model parameters
            X (jnp.ndarray): Input features
            y (jnp.ndarray): Target values
            
        Returns:
            float: Computed loss value
        """
        predictions = self.forward(params, X)
        return self.loss_function(y, predictions)
    
    def train(self, X, y):
        """
        Train the linear regression model using gradient descent.
        
        Args:
            X (jnp.ndarray): Input features
            y (jnp.ndarray): Target values
            
        Returns:
            dict: Fitted model parameters
        """
        # Initialize parameters
        current_params = self.init_params(X.shape[1])

        # Define the update step as a pure, JIT-compiled function
        @jax.jit
        def update_step(params, X, y):
            # Compute gradients
            grads = jax.grad(self.loss_fn)(params, X, y)

            return jax.tree_util.tree_map(
                lambda p, g: p - self.learning_rate * g, params, grads
            )

        # Training loop
        for epoch in range(self.n_epochs):
            # Get the new, updated parameters from the pure function
            current_params = update_step(current_params, X, y)

            if epoch % 100 == 0:
                loss_value = self.loss_fn(current_params, X, y)
                print(f"Epoch {epoch}, Loss: {loss_value}")
        # Store the final parameters and return the instance
        self.params = current_params
        return self
        

