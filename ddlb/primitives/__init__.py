"""
Distributed primitives for deep learning benchmarks
"""

from .SP_TP_Rowwise import SP_TP_Rowwise

__all__ = ['SP_TP_Rowwise', 'PyTorch_SP_TP_Rowwise']

# Re-export lazily to avoid importing CUDA-heavy backends unless needed
import importlib
import typing as _typing

if _typing.TYPE_CHECKING:
    from .SP_TP_Rowwise import PyTorch_SP_TP_Rowwise  # noqa: F401


def __getattr__(name):
    if name == 'PyTorch_SP_TP_Rowwise':
        return importlib.import_module('.SP_TP_Rowwise', __name__).PyTorch_SP_TP_Rowwise
    raise AttributeError(f"module {__name__} has no attribute {name}")