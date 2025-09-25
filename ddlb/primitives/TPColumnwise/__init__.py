"""
Tensor Parallel Column-wise primitive implementations
"""

from .tp_columnwise import TPColumnwise

__all__ = [
    'TPColumnwise',
    'PyTorchTPColumnwise',
    'ComputeOnlyTPColumnwise',
    'FuserTPColumnwise',
    'TransformerEngineTPColumnwise',
]

# Lazy attribute-based imports to avoid importing optional heavy deps (e.g.,
# nvfuser, transformer_engine) until the specific implementation is requested.
import importlib
import typing as _typing

if _typing.TYPE_CHECKING:  # for type checkers only; does not execute at runtime
    from .pytorch import PyTorchTPColumnwise  # noqa: F401
    from .compute_only import ComputeOnlyTPColumnwise  # noqa: F401
    from .fuser import FuserTPColumnwise  # noqa: F401
    from .transformer_engine import TransformerEngineTPColumnwise  # noqa: F401


def __getattr__(name):
    if name == 'PyTorchTPColumnwise':
        return importlib.import_module('.pytorch', __name__).PyTorchTPColumnwise
    if name == 'ComputeOnlyTPColumnwise':
        return importlib.import_module('.compute_only', __name__).ComputeOnlyTPColumnwise
    if name == 'FuserTPColumnwise':
        return importlib.import_module('.fuser', __name__).FuserTPColumnwise
    if name == 'TransformerEngineTPColumnwise':
        return importlib.import_module('.transformer_engine', __name__).TransformerEngineTPColumnwise
    raise AttributeError(f"module {__name__} has no attribute {name}")