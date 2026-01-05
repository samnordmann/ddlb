"""
Sparse Tensor Parallel Row-wise primitive implementations
"""

from .sp_tp_rowwise import SP_TP_Rowwise

__all__ = [
    'SP_TP_Rowwise',
    'PyTorchSP_TP_Rowwise',
    'ComputeOnlySP_TP_Rowwise',
    'FuserSP_TP_Rowwise',
    'TransformerEngineSP_TP_Rowwise',
    'JAXSP_TP_Rowwise',
]

# Lazy attribute-based imports to avoid importing optional heavy deps (e.g.,
# nvfuser, transformer_engine) until the specific implementation is requested.
import importlib
import typing as _typing

if _typing.TYPE_CHECKING:  # for type checkers only; does not execute at runtime
    from .pytorch import PyTorchSP_TP_Rowwise  # noqa: F401
    from .compute_only import ComputeOnlySP_TP_Rowwise  # noqa: F401
    from .fuser import FuserSP_TP_Rowwise  # noqa: F401
    from .transformer_engine import TransformerEngineSP_TP_Rowwise  # noqa: F401
    from .jax_tp import JAXSP_TP_Rowwise  # noqa: F401

def __getattr__(name):
    if name == 'PyTorchSP_TP_Rowwise':
        return importlib.import_module('.pytorch', __name__).PyTorchSP_TP_Rowwise
    if name == 'ComputeOnlySP_TP_Rowwise':
        return importlib.import_module('.compute_only', __name__).ComputeOnlySP_TP_Rowwise
    if name == 'FuserSP_TP_Rowwise':
        return importlib.import_module('.fuser', __name__).FuserSP_TP_Rowwise
    if name == 'TransformerEngineSP_TP_Rowwise':
        return importlib.import_module('.transformer_engine', __name__).TransformerEngineSP_TP_Rowwise
    if name == 'JAXSP_TP_Rowwise':
        return importlib.import_module('.jax_tp', __name__).JAXSP_TP_Rowwise
    raise AttributeError(f"module {__name__} has no attribute {name}")
