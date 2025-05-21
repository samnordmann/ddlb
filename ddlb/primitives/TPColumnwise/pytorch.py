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
    
    DEFAULT_OPTIONS = {
        'backend': 'nccl'  # Default backend
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Set backend from options or use default
        self.backend = kwargs.get('backend', self.DEFAULT_OPTIONS['backend'])
        
        # Create a new process group with specified backend
        ranks = list(range(self.communicator.world_size))
        self.pg = dist.new_group(ranks=ranks, backend=self.backend)
        
        # Pre-allocate tensor for allgather
        self.A_gathered = torch.empty(
            self.m,
            self.k,
            dtype=self.A.dtype,
            device=self.A.device
        )
    
    def run(self) -> torch.Tensor:
        """
        Run the TP Column-wise operation.
        
        Returns:
            The result matrix of shape (m, n)
        """
        # Allgather A from all GPUs using our specific group
        dist.all_gather_into_tensor(self.A_gathered, self.A, group=self.pg)
        
        # Compute matrix multiplication
        C = torch.matmul(self.A_gathered, self.B)
        
        return C
