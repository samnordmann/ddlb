"""
Tensor Parallel Row-wise primitive implementations with Sequence Parallelism
"""

from .tp_rowwise import TPRowwise

__all__ = [
    'TPRowwise',
    'PyTorchTPRowwise',
    'FuserTPRowwise',
    'TransformerEngineTPRowwise',
    'CustomKernelTPRowwise',
]

# Lazy attribute-based imports to avoid importing optional heavy deps until
# the specific implementation is requested.
import importlib
import typing as _typing

if _typing.TYPE_CHECKING:  # for type checkers only; does not execute at runtime
    from .pytorch import PyTorchTPRowwise  # noqa: F401
    from .fuser import FuserTPRowwise  # noqa: F401
    from .transformer_engine import TransformerEngineTPRowwise  # noqa: F401
    from .custom_kernel import CustomKernelTPRowwise  # noqa: F401

def __getattr__(name):
    if name == 'PyTorchTPRowwise':
        return importlib.import_module('.pytorch', __name__).PyTorchTPRowwise
    if name == 'TransformerEngineTPRowwise':
        return importlib.import_module('.transformer_engine', __name__).TransformerEngineTPRowwise
    if name == 'FuserTPRowwise':
        return importlib.import_module('.fuser', __name__).FuserTPRowwise
    if name == 'CustomKernelTPRowwise':
        return importlib.import_module('.custom_kernel', __name__).CustomKernelTPRowwise
    raise AttributeError(f"module {__name__} has no attribute {name}")

