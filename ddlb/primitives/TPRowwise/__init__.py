"""
Tensor Parallel Row-wise primitive implementations with Sequence Parallelism
"""

from .tp_rowwise import TPRowwise

__all__ = [
    'TPRowwise',
    'PyTorchTPRowwise',
]

# Lazy attribute-based imports to avoid importing optional heavy deps until
# the specific implementation is requested.
import importlib
import typing as _typing

if _typing.TYPE_CHECKING:  # for type checkers only; does not execute at runtime
    from .pytorch import PyTorchTPRowwise  # noqa: F401

def __getattr__(name):
    if name == 'PyTorchTPRowwise':
        return importlib.import_module('.pytorch', __name__).PyTorchTPRowwise
    raise AttributeError(f"module {__name__} has no attribute {name}")

