"""Shared foundations for linear-model estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
import time

import jax
import jax.nn
import jax.numpy as jnp

from jasmine.metrics import accuracy_score, r2_score


class BaseLinearModel(ABC):
    """Base class for gradient-based linear models."""

    center_targets = False
    log_every_epoch = False
    stop_on_invalid_loss = False

    def __init__(
        self,
        use_bias=True,
        learning_rate=0.01,
        n_epochs=1000,
        loss_function=None,
        l1_penalty=0.0,
        l2_penalty=0.0,
    ):
        self.use_bias = use_bias
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.loss_function = loss_function
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.params = None

    def _initialization_key(self):
        return jax.random.PRNGKey(42)

    def init_params(self, n_features, key=None):
        """Initialize model weights and an optional bias term."""
        if key is None:
            key = self._initialization_key()

        w_key, _ = jax.random.split(key)
        params = {"w": jax.random.normal(w_key, (n_features,))}
        params["b"] = jnp.array(0.0) if self.use_bias else None

        self.params = params
        return params

    @staticmethod
    def forward(params, X):
        """Compute the linear response for the provided parameters."""
        if params.get("w") is None:
            raise ValueError("Model weights must be initialized before calling `forward`.")

        outputs = X @ params["w"]
        if params.get("b") is not None:
            outputs = outputs + params["b"]

        return outputs

    def _regularization_penalty(self, params):
        penalty = 0.0

        if self.l2_penalty > 0.0:
            penalty += self.l2_penalty * jnp.sum(params["w"] ** 2)

        if self.l1_penalty > 0.0:
            penalty += self.l1_penalty * jnp.sum(jnp.abs(params["w"]))

        return penalty

    def _require_fitted(self, method_name):
        if self.params is None:
            raise ValueError(
                f"Model has not been trained yet. Call `train` before calling `{method_name}`."
            )

    def _preprocess_training_data(self, X, y):
        """Center features, and optionally targets, when fitting a bias term."""
        X = jnp.asarray(X)
        y = jnp.asarray(y)

        X_offset = jnp.zeros(X.shape[1], dtype=X.dtype)
        y_offset = jnp.array(0.0, dtype=X.dtype)

        if not self.use_bias:
            return X, y, X_offset, y_offset

        X_offset = jnp.mean(X, axis=0)
        X_processed = X - X_offset

        if not self.center_targets:
            return X_processed, y, X_offset, y_offset

        y_processed = jnp.asarray(y, dtype=X.dtype)
        y_offset = jnp.mean(y_processed, axis=0)
        return X_processed, y_processed - y_offset, X_offset, y_offset

    def _apply_training_offsets(self, X, y, X_offset, y_offset):
        """Apply training-time centering offsets to another dataset."""
        X = jnp.asarray(X)
        y = jnp.asarray(y)

        if self.use_bias:
            X = X - X_offset

        if self.center_targets:
            y = jnp.asarray(y, dtype=X.dtype) - y_offset

        return X, y

    def _set_intercept(self, params, X_offset, y_offset=0.0):
        """Translate a centered-space bias term back to raw input space."""
        params = dict(params)
        dtype = params["w"].dtype

        if not self.use_bias:
            self.intercept_ = jnp.array(0.0, dtype=dtype)
            return params

        centered_bias = params.get("b")
        if centered_bias is None:
            centered_bias = jnp.array(0.0, dtype=dtype)
        else:
            centered_bias = jnp.asarray(centered_bias, dtype=dtype)

        X_offset = jnp.asarray(X_offset, dtype=dtype)
        y_offset = jnp.asarray(y_offset, dtype=dtype)
        intercept = centered_bias + y_offset - X_offset @ params["w"]

        params["b"] = intercept
        self.intercept_ = intercept
        return params

    @staticmethod
    def _validate_training_data(X, y):
        if X.ndim != 2:
            raise ValueError(f"Input features X must be a 2D array, got shape {X.shape}.")
        if y.ndim != 1:
            raise ValueError(f"Target values y must be a 1D array, got shape {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Number of samples in X ({X.shape[0]}) must match number of samples in y ({y.shape[0]})."
            )

    @classmethod
    def _validate_validation_data(cls, validation_data, n_features):
        if validation_data is None:
            return None

        X_val, y_val = validation_data
        cls._validate_training_data(X_val, y_val)

        if X_val.shape[1] != n_features:
            raise ValueError(
                f"Validation features must have {n_features} columns, got {X_val.shape[1]}."
            )

        return X_val, y_val

    def _make_update_step(self):
        @jax.jit
        def update_step(params, X, y):
            grads = jax.grad(self.loss_fn)(params, X, y)
            return jax.tree_util.tree_map(
                lambda param, grad: param - self.learning_rate * grad,
                params,
                grads,
            )

        return update_step

    @staticmethod
    def _format_log(epoch, n_epochs, train_loss, val_loss=None):
        message = f"Epoch {epoch}/{n_epochs} - Loss: {float(train_loss):.4f}"
        if val_loss is not None:
            message += f" - Val Loss: {float(val_loss):.4f}"
        return message

    @abstractmethod
    def loss_fn(self, params, X, y):
        """Compute the training loss for the estimator."""

    def train(self, X, y, validation_data=None, early_stopping_patience=None, verbose=1):
        """Train the estimator with gradient descent."""
        self._validate_training_data(X, y)
        validation_data = self._validate_validation_data(validation_data, X.shape[1])

        X_train, y_train, X_offset, y_offset = self._preprocess_training_data(X, y)
        self.X_offset_ = X_offset
        self.y_offset_ = y_offset

        if validation_data is not None:
            X_val, y_val = validation_data
            validation_data = self._apply_training_offsets(X_val, y_val, X_offset, y_offset)

        current_params = self.init_params(X_train.shape[1])
        history = {"loss": [], "val_loss": []}

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_params = None
        update_step = self._make_update_step()

        start_time = time.time()
        print_interval = 1 if self.log_every_epoch else max(1, self.n_epochs // 10)

        for epoch in range(self.n_epochs):
            current_params = update_step(current_params, X_train, y_train)
            train_loss = self.loss_fn(current_params, X_train, y_train)

            if self.stop_on_invalid_loss and bool(jnp.isnan(train_loss) | jnp.isinf(train_loss)):
                if verbose > 0:
                    print(f"Loss is NaN or Inf at epoch {epoch + 1}. Stopping training.")
                break

            history["loss"].append(train_loss)
            val_loss = None

            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss = self.loss_fn(current_params, X_val, y_val)
                history["val_loss"].append(val_loss)

                if early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                        best_params = current_params
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= early_stopping_patience:
                        if verbose > 0:
                            print(
                                f"Early stopping at epoch {epoch + 1}. Best Val Loss: {float(best_val_loss):.4f}"
                            )
                        final_params = best_params if best_params is not None else current_params
                        self.params = self._set_intercept(final_params, X_offset, y_offset)
                        return history

            should_log = (epoch + 1) % print_interval == 0 or epoch + 1 == self.n_epochs
            if verbose > 0 and should_log:
                print(self._format_log(epoch + 1, self.n_epochs, train_loss, val_loss), end="\r")

        if verbose > 0:
            total_time = time.time() - start_time
            print(f"\nTraining completed in {total_time:.2f} seconds.")

        final_params = best_params if best_params is not None else current_params
        self.params = self._set_intercept(final_params, X_offset, y_offset)
        return history


class RegressorMixin:
    """Shared inference and evaluation helpers for regressors."""

    def inference(self, X):
        self._require_fitted("inference")
        return self.forward(self.params, X)

    def evaluate(self, X, y, metrics_fn=r2_score):
        predictions = self.inference(X)
        return metrics_fn(y, predictions)


class BinaryClassifierMixin:
    """Shared helpers for binary linear classifiers."""

    def predict_probabilities(self, X):
        self._require_fitted("predict_probabilities")
        logits = self.forward(self.params, X)
        return jax.nn.sigmoid(logits)

    def inference(self, X, threshold=0.5):
        probabilities = self.predict_probabilities(X)
        return (probabilities >= threshold).astype(int)

    def evaluate(self, X, y, metrics_fn=accuracy_score):
        class_predictions = self.inference(X)
        return metrics_fn(y, class_predictions)


__all__ = ["BaseLinearModel", "BinaryClassifierMixin", "RegressorMixin"]
