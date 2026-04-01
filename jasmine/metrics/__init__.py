# Backward-compatible re-exports: these functions are also in jasmine.losses
from jasmine.losses._classification import bce_loss as binary_cross_entropy
from jasmine.losses._regression import mse_loss as mean_squared_error

from ._regression import mean_absolute_error, root_mean_squared_error, r2_score

from ._classification import (
    categorical_cross_entropy,
    accuracy_score,
    euclidean_distance,
    manhattan_distance,
)

__all__ = [
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "r2_score",
    "binary_cross_entropy",
    "categorical_cross_entropy",
    "accuracy_score",
    "euclidean_distance",
    "manhattan_distance",
]
