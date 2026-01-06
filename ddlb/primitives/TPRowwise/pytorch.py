"""
PyTorch implementation of TP Row-wise primitive with Sequence Parallelism
"""

import os
import torch
import torch.distributed as dist

from .tp_rowwise import TPRowwise
from .utils import EnvVarGuard, setup_ucc_env_vars
from ddlb.envs import get_master_addr, get_master_port

class PyTorchTPRowwise(TPRowwise):
    """
    PyTorch implementation of TP Row-wise primitive with Sequence Parallelism.
    
    Performs local matrix multiplication followed by Reduce-Scatter to sum partial results
    and shard the output along the sequence dimension (rows).
    
    Supports both NCCL and UCC backends. For UCC, transport layer can be specified
    using the format 'ucc/tl/<transport_layer>'.
    """
    
    DEFAULT_OPTIONS = {
        'backend': 'nccl',  # Default backend
    }
    
    ALLOWED_VALUES = {
        'backend': ['nccl', 'ucc', 'ucc/tl/nccl', 'ucc/tl/cuda', 'ucc/tl/ucp'],
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

        # Allocate space for the output shard
        chunk_size = self.m // self.communicator.world_size
        self.result_shard = torch.empty(
            chunk_size,
            self.n,
            dtype=self.A.dtype,
            device=self.A.device
        )

    def __del__(self):
        dist.destroy_process_group()
        self.communicator.barrier()

    def run(self) -> torch.Tensor:
        """
        Run the TP Row-wise operation with Sequence Parallelism.
        
        Performs local matmul followed by reduce-scatter to sum and shard results along rows.
        
        Returns:
            torch.Tensor: Result matrix of shape (m_local, n) where m_local = m // world_size
        """
        torch.cuda.empty_cache()

        # Perform local matmul, then reduce-scatter to sum and shard results along rows
        local_result = torch.matmul(self.A, self.B)
        dist.reduce_scatter_tensor(self.result_shard, local_result, op=dist.ReduceOp.SUM)
        
        return self.result_shard

