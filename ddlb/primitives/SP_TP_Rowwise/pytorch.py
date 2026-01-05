"""
PyTorch implementation of SP_TP_Rowwise primitive
"""

import os
import torch
import torch.distributed as dist

from .sp_tp_rowwise import SP_TP_Rowwise
from .utils import EnvVarGuard, setup_ucc_env_vars
from ddlb.envs import get_master_addr, get_master_port

class PyTorch_SP_TP_Rowwise(SP_TP_Rowwise):
    """
    PyTorch implementation of SP_TP_Rowwise primitive using PyTorch's distributed module.
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
            pytorch_backend = 'ucc'
        else:
            pytorch_backend = backend
        
        # Set up environment variables
        self.env_guard = EnvVarGuard(setup_ucc_env_vars(backend))
    
        # Parse DDLB prefixed environment variables with defaults
        master_addr = get_master_addr()
        master_port = get_master_port()

        dist.init_process_group(
            backend=pytorch_backend,
            rank=self.communicator.rank,
            world_size=self.communicator.world_size,
            init_method=f"tcp://{master_addr}:{master_port}",
            device_id=self.communicator.device
        )

        # Get allgather order
        self.order = self.options['order']

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
        dist.destroy_process_group()
        self.communicator.barrier()

    def run(self) -> torch.Tensor:
        """
        Run the SP_TP_Rowwise operation.
        
        Returns:
            torch.Tensor: Result matrix of shape (m, n)
        """
        torch.cuda.empty_cache()

        if self.order == 'AG_before':
            # First allgather A, then do matmul
            dist.all_gather_into_tensor(self.A_gathered, self.A)
            result = torch.matmul(self.A_gathered, self.B)
        else:
            # First do local matmul, then allgather results
            local_result = torch.matmul(self.A, self.B)
            dist.all_gather_into_tensor(self.result_gathered, local_result)
            result = self.result_gathered
        
        return result

