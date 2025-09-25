"""
Distributed primitives for deep learning benchmarks
"""

from .TPColumnwise import TPColumnwise

__all__ = ['TPColumnwise', 'PyTorchTPColumnwise']

# Re-export lazily to avoid importing CUDA-heavy backends unless needed
import importlib
import typing as _typing

if _typing.TYPE_CHECKING:
    from .TPColumnwise import PyTorchTPColumnwise  # noqa: F401


def __getattr__(name):
    if name == 'PyTorchTPColumnwise':
        return importlib.import_module('.TPColumnwise', __name__).PyTorchTPColumnwise
    raise AttributeError(f"module {__name__} has no attribute {name}")