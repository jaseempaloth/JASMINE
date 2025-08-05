import jax
import jax.numpy as jnp
from jasmine.preprocessing.scaler import StandardScaler

#. Generate some random data
key = jax.random.PRNGKey(42)
X_train = jax.random.normal(key, (100000, 100)) * 10 + 5 # (n_samples, n_features)

# 2. Create and fit the new JAX-optimized scaler
scaler = StandardScaler()
scaler.fit(X_train)

# 3. Prepare some test data
key, subkey = jax.random.split(key)
X_test = jax.random.normal(subkey, (20000, 100)) * 10 + 5

# 4. Time the transformation
print("Timing the JIT-compiled transform function:")

# The first call to transform will be a bit slower because JAX is compiling the function.
# We use .block_until_ready() to get accurate timing.
print("First call (includes compilation time):")
import timeit
first_call_time = timeit.timeit(lambda: scaler.transform(X_test).block_until_ready(), number=1)
print(f"First call time: {first_call_time:.6f} seconds")

# Subsequent calls will be much faster because they use the cached, compiled code.
print("\nSecond call (uses cached compiled code):")
import timeit
second_call_time = timeit.timeit(lambda: scaler.transform(X_test).block_until_ready(), number=1)
print(f"Second call time: {second_call_time:.6f} seconds")

# 5. Verify the results
X_transformed = scaler.transform(X_test)
X_reverted = scaler.inverse_transform(X_transformed)

print("\n--- Verification ---")
print(f"Original X_test mean (first 5 features): {jnp.mean(X_test, axis=0)[:5]}")
print(f"Transformed X_test mean (should be ~0): {jnp.mean(X_transformed, axis=0)[:5]}")
print(f"Reverted X_test mean (should be ~original): {jnp.mean(X_reverted, axis=0)[:5]}")