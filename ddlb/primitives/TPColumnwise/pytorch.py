"""
PyTorch implementation of TP Column-wise primitive
"""

import os
import torch
import torch.distributed as dist

from .tp_columnwise import TPColumnwise

class PyTorchTPColumnwise(TPColumnwise):
    """
    PyTorch implementation of TP Column-wise primitive using PyTorch's distributed module.
    Performs Allgather on A followed by matrix multiplication with B.
    
    Supports both NCCL and UCC backends. For UCC, transport layer can be specified
    using the format 'ucc/tl/<transport_layer>'.
    """
    
    DEFAULT_OPTIONS = {
        'backend': 'nccl'  # Default backend
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Parse backend configuration
        backend = kwargs.get('backend', self.DEFAULT_OPTIONS['backend'])
        
        if backend.startswith('ucc/tl/'):
            self.tl = backend.split('/')[-1]
            os.environ["UCC_CL_BASIC_TLS"] = self.tl
            self.backend = 'ucc'
        else:
            self.backend = backend
            self.tl = None
        
        # Initialize process group and allocate tensors
        ranks = list(range(self.communicator.world_size))
        self.pg = dist.new_group(ranks=ranks, backend=self.backend)
        self.A_gathered = torch.empty(
            self.m,
            self.k,
            dtype=self.A.dtype,
            device=self.A.device
        )
    
    def __del__(self):
        """Clean up process group and environment variables."""
        if self.tl is not None and "UCC_CL_BASIC_TLS" in os.environ:
            del os.environ["UCC_CL_BASIC_TLS"]
        
        if hasattr(self, 'pg'):
            dist.destroy_process_group(self.pg)
    
    def run(self) -> torch.Tensor:
        """
        Run the TP Column-wise operation.
        
        Returns:
            torch.Tensor: Result matrix of shape (m, n)
        """
        dist.all_gather_into_tensor(self.A_gathered, self.A, group=self.pg)
        return torch.matmul(self.A_gathered, self.B)

