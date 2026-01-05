"""
Distributed Deep Learning Benchmark (DDLB)
"""

__version__ = "0.1.0"

# Avoid importing heavy dependencies at package import time. Expose symbols lazily.
import importlib
import typing as _typing

__all__ = [
    "PrimitiveBenchmarkRunner",
    "SP_TP_Rowwise",
    "PyTorchSP_TP_Rowwise",
]

if _typing.TYPE_CHECKING:
    from .benchmark import PrimitiveBenchmarkRunner  # noqa: F401
    from .primitives import SP_TP_Rowwise, PyTorchSP_TP_Rowwise  # noqa: F401


def __getattr__(name):
    if name == 'PrimitiveBenchmarkRunner':
        return importlib.import_module('.benchmark', __name__).PrimitiveBenchmarkRunner
    if name == 'SP_TP_Rowwise' or name == 'PyTorchSP_TP_Rowwise':
        return getattr(importlib.import_module('.primitives', __name__), name)
    raise AttributeError(f"module {__name__} has no attribute {name}")