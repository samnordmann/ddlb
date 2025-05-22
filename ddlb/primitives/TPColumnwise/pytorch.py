"""
PyTorch implementation of TP Column-wise primitive
"""

import os
import torch
import torch.distributed as dist

from .tp_columnwise import TPColumnwise
from .utils import EnvVarGuard, setup_ucc_env_vars

class PyTorchTPColumnwise(TPColumnwise):
    """
    PyTorch implementation of TP Column-wise primitive using PyTorch's distributed module.
    Performs Allgather on A followed by matrix multiplication with B.
    
    Supports both NCCL and UCC backends. For UCC, transport layer can be specified
    using the format 'ucc/tl/<transport_layer>'.
    
    The order of operations can be controlled using the 'order' option:
    - 'AG_before': First perform allgather, then matmul (default)
    - 'AG_after': First perform local matmul, then allgather results
    """
    
    DEFAULT_OPTIONS = {
        'backend': 'nccl',  # Default backend
        'order': 'AG_before'  # Default order
    }
    
    ALLOWED_VALUES = {
        'backend': ['nccl', 'ucc', 'ucc/tl/nccl', 'ucc/tl/cuda', 'ucc/tl/ucp'],
        'order': ['AG_before', 'AG_after']
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Parse backend configuration
        backend = self.options['backend']
        
        if backend.startswith('ucc/tl/'):
            self.tl = backend.split('/')[-1]
            self.backend = 'ucc'
        else:
            self.backend = backend
            self.tl = None
        
        if backend == 'nccl':
            # Do a dummy allreduce to avoid hang later
            dummy = torch.ones(1, device=self.communicator.device)
            dist.all_reduce(dummy)

        # Set up environment variables
        self.env_guard = EnvVarGuard(setup_ucc_env_vars(backend))
    
        # Get allgather order
        self.order = self.options['order']
        
        # Initialize process group and allocate tensors
        ranks = list(range(self.communicator.world_size))
        self.pg = dist.new_group(ranks=ranks, backend=self.backend, device_id=self.communicator.device)

        if self.order == 'AG_before':
            # For AG_before, we need space for the full A matrix
            self.A_gathered = torch.empty(
                self.m,
                self.k,
                dtype=self.A.dtype,
                device=self.A.device
            )
        else:
            # For AG_after, we need space for the full result matrix
            self.result_gathered = torch.empty(
                self.m,
                self.n,
                dtype=self.A.dtype,
                device=self.A.device
            )
    
    def __del__(self):
        """Clean up process group"""
        if hasattr(self, 'pg'):
            dist.barrier()
            dist.destroy_process_group(self.pg)
            dist.barrier()
    
    def run(self) -> torch.Tensor:
        """
        Run the TP Column-wise operation.
        
        Returns:
            torch.Tensor: Result matrix of shape (m, n)
        """
        # Synchronize processes before starting
        self.communicator.barrier()
        torch.cuda.synchronize()
        
        if self.order == 'AG_before':
            # First allgather A, then do matmul
            dist.all_gather_into_tensor(self.A_gathered, self.A, group=self.pg)
            result = torch.matmul(self.A_gathered, self.B)
        else:
            # First do local matmul, then allgather results
            local_result = torch.matmul(self.A, self.B)
            dist.all_gather_into_tensor(self.result_gathered, local_result, group=self.pg)
            result = self.result_gathered
        
        # Synchronize processes after completion
        self.communicator.barrier()
        torch.cuda.synchronize()
        
        return result

