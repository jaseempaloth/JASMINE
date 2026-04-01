import jax.numpy as jnp
import numpy as np

from jasmine.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from jasmine.losses import bce_loss, hinge_loss, mse_loss
from jasmine.neighbors import KNNClassifier
from jasmine.optim import Adam, Momentum, SGD
from jasmine.svm import SVMClassifier
from jasmine.model_selection import train_test_split
from jasmine.preprocessing import OneHotEncoder, StandardScaler
from jasmine.datasets import generate_regression
from jasmine import (
    ElasticNet as PublicElasticNet,
    Lasso as PublicLasso,
    LinearRegression as PublicLinearRegression,
    LogisticRegression as PublicLogisticRegression,
    Ridge as PublicRidge,
    KNNClassifier as PublicKNNClassifier,
    SVMClassifier as PublicSVMClassifier,
    train_test_split as PublicTrainTestSplit,
    SGD as PublicSGD,
    Momentum as PublicMomentum,
    Adam as PublicAdam,
    mse_loss as PublicMseLoss,
    bce_loss as PublicBceLoss,
    hinge_loss as PublicHingeLoss,
)


def test_public_api_exports_match_canonical_modules():
    assert PublicElasticNet is ElasticNet
    assert PublicLasso is Lasso
    assert PublicLinearRegression is LinearRegression
    assert PublicLogisticRegression is LogisticRegression
    assert PublicRidge is Ridge
    assert PublicKNNClassifier is KNNClassifier
    assert PublicSVMClassifier is SVMClassifier
    assert PublicTrainTestSplit is train_test_split
    assert PublicSGD is SGD
    assert PublicMomentum is Momentum
    assert PublicAdam is Adam
    assert PublicMseLoss is mse_loss
    assert PublicBceLoss is bce_loss
    assert PublicHingeLoss is hinge_loss


def test_dataset_split_and_preprocessing_components():
    X, y = generate_regression(n_samples=50, n_features=8, n_informative=4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    assert X.shape == (50, 8)
    assert y.shape == (50,)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    X_restored = scaler.inverse_transform(X_scaled)

    assert X_scaled.shape == X_train.shape
    assert jnp.allclose(X_restored, X_train, atol=1e-6)

    categories = np.array([["red", "small"], ["blue", "small"], ["red", "large"]])
    encoded = OneHotEncoder().fit_transform(categories)
    assert encoded.shape[0] == categories.shape[0]
