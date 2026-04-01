"""Abstract base class for JASMINE optimizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from jasmine._typing import OptState, Params


class BaseOptimizer(ABC):
    """Interface all JASMINE optimizers must satisfy."""

    @abstractmethod
    def init(self, params: Params) -> OptState:
        """Return an initial optimizer state compatible with ``params``."""

    @abstractmethod
    def update(self, grads: Params, state: OptState) -> Tuple[Params, OptState]:
        """Compute parameter updates and the next optimizer state."""
