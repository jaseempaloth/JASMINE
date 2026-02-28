Performance Notes
=================

JASMINE uses JAX JIT compilation for training and inference paths.

Tips
----

* First execution includes compile overhead.
* Use feature scaling for faster convergence.
* Keep tensor shapes stable across calls to maximize JIT cache reuse.
