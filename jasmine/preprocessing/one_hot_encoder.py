import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Dict, List, Union, Any
from functools import partial

EncoderParams = Dict[str, Any]

def fit_fn(X: jnp.ndarray, categories: Union[str, List[jnp.ndarray]] = 'auto') -> EncoderParams:
    """
    Pure function to learn the categories for one-hot encoding.
    
    Note: This function is not JIT-compiled because jnp.unique is not efficient
    for JIT compilation in this context.

    Args:
        X (jnp.ndarray): The data to determine the categories of.
        categories (Union[str, List[jnp.ndarray]]): The categories to use for encoding.
            If 'auto', the categories are inferred from the data.
            If a list, the categories are used as is.
        
    Returns:
        A dictionary containing the 'categories' for each feature.
    """
    if categories == 'auto':
        # Use standard NumPy for object arrays (strings), JAX NumPy for numeric arrays.
        # This makes the function robust to different data types.
        if X.dtype.kind in ["U", "S", "O"]:
            unique_fn = np.unique
        else:
            unique_fn = jnp.unique
        # Learn categories from the data for each feature
        learned_categories = [unique_fn(X[:, i]) for i in range(X.shape[1])]
    else:
        learned_categories = categories
    
    return {'categories': learned_categories}

@partial(jax.jit)
def transform_fn(X: jnp.ndarray, params: EncoderParams, dtype: jnp.dtype) -> jnp.ndarray:
    """
    Pure, JIT-compiled function to transform data using one-hot encoding.

    Args:
        X (jnp.ndarray): The data to transform.
        params (EncoderParams): The learned categories for each feature.
        dtype (jnp.dtype): The data type for the output array.
        
    Returns:
        jnp.ndarray: The one-hot encoded data.
    """
    output_arrays = []
    for i, cats in enumerate(params['categories']):
        feature_column = X[:, i]
        # The comparison (feature_col[:, None] == cats) creates a boolean matrix
        # which is then cast to the desired dtype.
        one_hot_matrix = (feature_column[:, None] == cats).astype(dtype)
        output_arrays.append(one_hot_matrix)
    
    return jnp.concatenate(output_arrays, axis=1)

@partial(jax.jit)
def inverse_transform_fn(X: jnp.ndarray, params: EncoderParams) -> jnp.ndarray:
    """
    Pure function to inverse transform one-hot encoded data back to original categories.
    
    Args:
        X (jnp.ndarray): The one-hot encoded data to inverse transform.
        params (EncoderParams): The learned categories for each feature.
        
    Returns:
        jnp.ndarray: The original data before one-hot encoding.
    """
    inverse_transformed = []
    start_idx = 0
    for cats in params['categories']:
        n_cats = len(cats)
        end_idx = start_idx + n_cats
        one_hot_encoded = X[:, start_idx:end_idx]
        original_feature = jnp.argmax(one_hot_encoded, axis=1)
        inverse_transformed.append(original_feature)
        start_idx = end_idx
    
    return jnp.column_stack(inverse_transformed)

class OneHotEncoder:
    """
    Encode categorical features as a one-hot numeric array.

    This class acts as a stateful wrapper around pure, JIT-compatible functions.

    Args:
        categories (Union[str, List[jnp.ndarray]]): The categories to use for encoding.
            If 'auto', the categories are inferred from the data.
            If a list, the categories are used as is.
        handle_unknown (str): Whether to raise an error or ignore if an unknown category is found.
        dtype (jnp.dtype): The data type for the output array.
        
    """
    def __init__(self,
            categories: Union[str, List[jnp.ndarray]] = 'auto',
            handile_unknown: str = 'error',
            dtype: jnp.dtype = jnp.float32):
        
        if handile_unknown not in ['error', 'ignore']:
            raise ValueError("`handle_unknown` must be 'error' or 'ignore'")
        
        self.categories = categories
        self.handle_unknown = handile_unknown
        self.dtype = dtype
        self.params: Optional[EncoderParams] = None

    def fit(self, X: jnp.ndarray):
        """
        Fit the encoder to the data.
        
        Args:
            X (jnp.ndarray): Input features of shape (n_samples, n_features).
        
        Returns:
            self: Fitted encoder instance.
        """
        self.params = fit_fn(X, self.categories)
        return self
    
    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Transform the data using the fitted encoder.
        
        Args:
            X (jnp.ndarray): Input features of shape (n_samples, n_features).
        
        Returns:
            jnp.ndarray: One-hot encoded features.
        """
        if self.params is None:
            raise RuntimeError("OneHotEncoder is not fitted yet. Call 'fit' first.")
        
        if self.handle_unknown == 'error':
            # Perform the check for unknown categories outside the JIT-compiled function.
            for i, cats in enumerate(self.params['categories']):
                # Find unique values in the column to be transformed
                unique_values = jnp.unique(X[:, i])
                # Check if any of these unique values are not in the learned categories
                is_unknown = ~jnp.isin(unique_values, cats)
                if jnp.any(is_unknown):
                    unknown_value = unique_values[jnp.argmax(is_unknown)]
                    raise ValueError(f"Found unknown category {unknown_value} in feature {i} during transform.")
        
        return transform_fn(X, self.params, self.dtype)
    
    def fit_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Fit the encoder to the data and transform it in one step.
        
        Args:
            X (jnp.ndarray): Input features of shape (n_samples, n_features).
        
        Returns:
            jnp.ndarray: One-hot encoded features.
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Inverse transform the one-hot encoded data back to original categories.
        
        Args:
            X (jnp.ndarray): One-hot encoded features of shape (n_samples, n_features).
        
        Returns:
            jnp.ndarray: Original features.
        """
        if self.params is None:
            raise RuntimeError("OneHotEncoder is not fitted yet. Call 'fit' first.")
        
        return inverse_transform_fn(X, self.params)
    
    


