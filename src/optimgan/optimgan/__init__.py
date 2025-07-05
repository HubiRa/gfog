from .default import DefaultOptimGan
from .wasserstein import WassersteinOptimGan
from .random import RandomOpt

__all__ = ["DefaultOptimGan", "WassersteinOptimGan", "RandomOpt"]
