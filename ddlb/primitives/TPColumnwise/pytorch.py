"""
PyTorch implementation of TP Column-wise primitive
"""

import torch
import torch.distributed as dist

from .tp_columnwise import TPColumnwise

class PyTorchTPColumnwise(TPColumnwise):
    """
    PyTorch implementation of TP Column-wise primitive using PyTorch's distributed module.
    Performs Allgather on A followed by matrix multiplication with B.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Pre-allocate tensors for allgather
        self.A_gathered = [torch.empty_like(self.A) for _ in range(self.communicator.world_size)]
    
    def run(self) -> torch.Tensor:
        """
        Run the TP Column-wise operation.
        
        Returns:
            The result matrix of shape (m, n)
        """
        # Allgather A from all GPUs
        dist.all_gather(self.A_gathered, self.A)
        
        # Concatenate the gathered pieces
        A_full = torch.cat(self.A_gathered, dim=1)
        
        # Compute matrix multiplication
        C = torch.matmul(A_full, self.B)
        
        return C
