"""
Tensor Parallel Column-wise primitive implementations
"""

from .tp_columnwise import TPColumnwise
from .pytorch_tp_columnwise import PyTorchTPColumnwise

__all__ = ['TPColumnwise', 'PyTorchTPColumnwise'] 