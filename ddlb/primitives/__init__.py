"""
Distributed primitives for deep learning benchmarks
"""

from .TPColumnwise import TPColumnwise, PyTorchTPColumnwise

__all__ = ['TPColumnwise', 'PyTorchTPColumnwise'] 