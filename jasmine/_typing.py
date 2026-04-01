"""Shared type aliases used across JASMINE submodules."""

from __future__ import annotations

from typing import Any, Dict

import jax

# Core array and parameter types
Array = jax.Array
Params = Dict[str, jax.Array]
PRNGKey = jax.Array
OptState = Dict[str, Any]
