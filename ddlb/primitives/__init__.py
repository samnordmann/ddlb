"""
Distributed primitives for deep learning benchmarks
"""

from .TPColumnwise import TPColumnwise
from .TPRowwise import TPRowwise

__all__ = ['TPColumnwise', 'PyTorchTPColumnwise', 'TPRowwise', 'PyTorchTPRowwise']

# Re-export lazily to avoid importing CUDA-heavy backends unless needed
import importlib
import typing as _typing

if _typing.TYPE_CHECKING:
    from .TPColumnwise import PyTorchTPColumnwise  # noqa: F401
    from .TPRowwise import PyTorchTPRowwise  # noqa: F401


def __getattr__(name):
    if name == 'PyTorchTPColumnwise':
        return importlib.import_module('.TPColumnwise', __name__).PyTorchTPColumnwise
    if name == 'PyTorchTPRowwise':
        return importlib.import_module('.TPRowwise', __name__).PyTorchTPRowwise
    raise AttributeError(f"module {__name__} has no attribute {name}")