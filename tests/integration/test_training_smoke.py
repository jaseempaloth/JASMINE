import jax.numpy as jnp

from jasmine.datasets import generate_classification, generate_regression
from jasmine.linear_model import LinearRegression, LogisticRegression
from jasmine.model_selection import train_test_split
from jasmine.neighbors import KNNClassifier
from jasmine.svm import SVMClassifier


def test_model_training_and_inference_smoke():
    X_reg, y_reg = generate_regression(
        n_samples=40,
        n_features=6,
        n_informative=3,
        random_state=21,
    )
    Xr_train, Xr_test, yr_train, _ = train_test_split(X_reg, y_reg, test_size=0.25, random_state=21)

    reg = LinearRegression(n_epochs=2, learning_rate=0.01)
    reg_history = reg.train(Xr_train, yr_train, verbose=0)
    assert len(reg_history["loss"]) >= 1
    assert reg.inference(Xr_test).shape[0] == Xr_test.shape[0]

    X_cls, y_cls = generate_classification(
        n_samples=40,
        n_features=6,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=8,
    )
    Xc_train, Xc_test, yc_train, _ = train_test_split(X_cls, y_cls, test_size=0.25, random_state=8)

    clf = LogisticRegression(n_epochs=2, learning_rate=0.01)
    clf_history = clf.train(Xc_train, yc_train, verbose=0)
    assert len(clf_history["loss"]) >= 1
    assert clf.inference(Xc_test).shape[0] == Xc_test.shape[0]

    knn = KNNClassifier(n_neighbors=3).train(Xc_train, yc_train)
    assert knn.inference(Xc_test).shape[0] == Xc_test.shape[0]

    y_svm = jnp.where(yc_train == 0, -1, 1)
    svm = SVMClassifier(n_epochs=2, learning_rate=0.01)
    svm_history = svm.train(Xc_train, y_svm, verbose=0)
    assert len(svm_history["loss"]) >= 1
    assert svm.inference(Xc_test).shape[0] == Xc_test.shape[0]
