"""
Sparse Tensor Parallel Row-wise primitive implementations
"""

from .sp_tp_rowwise import SP_TP_Rowwise

__all__ = [
    'SP_TP_Rowwise',
    'PyTorch_SP_TP_Rowwise',
    'ComputeOnly_SP_TP_Rowwise',
    'Fuser_SP_TP_Rowwise',
    'TransformerEngine_SP_TP_Rowwise',
    'JAX_SP_TP_Rowwise',
]

# Lazy attribute-based imports to avoid importing optional heavy deps (e.g.,
# nvfuser, transformer_engine) until the specific implementation is requested.
import importlib
import typing as _typing

if _typing.TYPE_CHECKING:  # for type checkers only; does not execute at runtime
    from .pytorch import PyTorch_SP_TP_Rowwise  # noqa: F401
    from .compute_only import ComputeOnly_SP_TP_Rowwise  # noqa: F401
    from .fuser import Fuser_SP_TP_Rowwise  # noqa: F401
    from .transformer_engine import TransformerEngine_SP_TP_Rowwise  # noqa: F401
    from .jax_tp import JAX_SP_TP_Rowwise  # noqa: F401

def __getattr__(name):
    if name == 'PyTorch_SP_TP_Rowwise':
        return importlib.import_module('.pytorch', __name__).PyTorch_SP_TP_Rowwise
    if name == 'ComputeOnly_SP_TP_Rowwise':
        return importlib.import_module('.compute_only', __name__).ComputeOnly_SP_TP_Rowwise
    if name == 'Fuser_SP_TP_Rowwise':
        return importlib.import_module('.fuser', __name__).Fuser_SP_TP_Rowwise
    if name == 'TransformerEngine_SP_TP_Rowwise':
        return importlib.import_module('.transformer_engine', __name__).TransformerEngine_SP_TP_Rowwise
    if name == 'JAX_SP_TP_Rowwise':
        return importlib.import_module('.jax_tp', __name__).JAX_SP_TP_Rowwise
    raise AttributeError(f"module {__name__} has no attribute {name}")
