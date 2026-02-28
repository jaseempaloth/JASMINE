FAQ
===

Why does the first call run slower?
-----------------------------------

JAX compiles functions on first run; subsequent calls reuse compiled kernels.

Does JASMINE support GPU?
-------------------------

Yes. If your JAX installation is CUDA/TPU-enabled, JASMINE runs on available accelerators.
