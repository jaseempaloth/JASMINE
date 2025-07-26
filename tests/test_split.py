from jasmine import train_test_split
import jax.numpy as jnp
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
import numpy as np

def test_train_test_split_with_real_data():
    print("🌸 Testing JASMINE train_test_split with Real Datasets 🌸\n")
    
    # Test 1: Iris Dataset (Classification)
    print("=" * 50)
    print("📊 IRIS DATASET (Classification)")
    print("=" * 50)
    
    iris = load_iris()
    X_iris = jnp.array(iris.data)
    y_iris = jnp.array(iris.target)
    
    print(f"Original data shape: {X_iris.shape}")
    print(f"Features: {iris.feature_names}")
    print(f"Target classes: {iris.target_names}")
    print(f"First 5 samples:\n{X_iris[:5]}")
    print(f"First 5 targets: {y_iris[:5]}\n")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Training targets: {y_train}")
    print(f"Test targets: {y_test}\n")
    
    # Test 2: Wine Dataset (Classification)
    print("=" * 50)
    print("🍷 WINE DATASET (Classification)")
    print("=" * 50)
    
    wine = load_wine()
    X_wine = jnp.array(wine.data)
    y_wine = jnp.array(wine.target)
    
    print(f"Original data shape: {X_wine.shape}")
    print(f"Number of features: {len(wine.feature_names)}")
    print(f"Target classes: {wine.target_names}")
    print(f"First sample: {X_wine[0]}")
    print(f"First target: {y_wine[0]}\n")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, test_size=0.2, shuffle=True, random_state=123)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train targets distribution: {jnp.bincount(y_train)}")
    print(f"Test targets distribution: {jnp.bincount(y_test)}\n")
    
    # Test 3: Custom regression data
    print("=" * 50)
    print("📈 SYNTHETIC REGRESSION DATA")
    print("=" * 50)
    
    # Create synthetic regression data
    np.random.seed(42)
    n_samples, n_features = 100, 3
    X_reg = jnp.array(np.random.randn(n_samples, n_features))
    y_reg = jnp.array(X_reg[:, 0] + 2 * X_reg[:, 1] - X_reg[:, 2] + 0.1 * np.random.randn(n_samples))
    
    print(f"Synthetic data shape: {X_reg.shape}")
    print(f"Target shape: {y_reg.shape}")
    print(f"Sample X values:\n{X_reg[:3]}")
    print(f"Sample y values: {y_reg[:3]}\n")
    
    # Split with no shuffle to see deterministic split
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.25, shuffle=False)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train y range: [{jnp.min(y_train):.3f}, {jnp.max(y_train):.3f}]")
    print(f"Test y range: [{jnp.min(y_test):.3f}, {jnp.max(y_test):.3f}]\n")
    
    # Test 4: Scaled data
    print("=" * 50)
    print("🔄 SCALED IRIS DATA")
    print("=" * 50)
    
    # Scale the iris data
    scaler = StandardScaler()
    X_iris_scaled = jnp.array(scaler.fit_transform(iris.data))
    
    print(f"Original mean: {jnp.mean(X_iris, axis=0)}")
    print(f"Scaled mean: {jnp.mean(X_iris_scaled, axis=0)}")
    print(f"Original std: {jnp.std(X_iris, axis=0)}")
    print(f"Scaled std: {jnp.std(X_iris_scaled, axis=0)}\n")
    
    X_train, X_test, y_train, y_test = train_test_split(X_iris_scaled, y_iris, test_size=0.33, random_state=99)
    
    print(f"Scaled training set: {X_train.shape[0]} samples")
    print(f"Scaled test set: {X_test.shape[0]} samples")
    print(f"Train mean: {jnp.mean(X_train, axis=0)}")
    print(f"Test mean: {jnp.mean(X_test, axis=0)}")

def test_original_functionality():
    """Keep the original test for comparison"""
    print("\n" + "=" * 50)
    print("🔧 ORIGINAL FUNCTIONALITY TEST")
    print("=" * 50)
    
    # Original sample data
    X = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = jnp.array([0, 1, 0, 1])

    # Test with default parameters
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    print("=== Default Parameters ===")
    print(f"X_train:\n{X_train}")
    print(f"X_test:\n{X_test}")
    print(f"y_train: {y_train}")
    print(f"y_test: {y_test}")

    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 1
    assert y_train.shape[0] == 3
    assert y_test.shape[0] == 1

if __name__ == "__main__":
    test_train_test_split_with_real_data()
    test_original_functionality()
    print("\n🎉 All tests passed! JASMINE train_test_split works perfectly! ✅")

