"""
nvFuser implementation of TP Column-wise primitive
"""

import os
import torch
import torch.distributed as dist
from nvfuser import FusionDefinition

from .tp_columnwise import TPColumnwise

class FuserTPColumnwise(TPColumnwise):
    """
    nvFuser implementation of TP Column-wise primitive.
    
    This implementation uses NVIDIA's nvFuser library to optimize the matrix multiplication
    operation. The fusion is done at the CUDA kernel level, which can provide better
    performance than standard PyTorch operations.
    
    The implementation supports both NCCL and UCC backends, similar to the PyTorch implementation.
    """
    
    DEFAULT_OPTIONS = {
        'backend': 'nccl',  # Default backend
        'order': 'AG_before',  # Default order
        'fusion_strategy': 'aggressive'  # Fusion strategy for nvFuser
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Parse backend configuration
        backend = kwargs.get('backend', self.DEFAULT_OPTIONS['backend'])
        
        self.env_vars = {}
        if backend.startswith('ucc/tl/'):
            self.tl = backend.split('/')[-1]
            self.env_vars["UCC_CL_BASIC_TLS"] = self.tl
            self.backend = 'ucc'
        else:
            if backend not in ['ucc', 'nccl']:
                raise ValueError(f"Invalid backend: {backend}. Must be 'ucc' or 'nccl'")
            self.backend = backend
            self.tl = None
        
        if self.backend == 'ucc':
            self.env_vars["UCX_RNDV_THRESH"] = "0"
            self.env_vars["UCX_TLS"] = "ib,cuda_copy"

        # Set environment variables
        for key, value in self.env_vars.items():
            os.environ[key] = value
    
        # Get allgather order
        self.order = kwargs.get('order', self.DEFAULT_OPTIONS['order'])
        if self.order not in ['AG_before', 'AG_after']:
            raise ValueError(f"Invalid order: {self.order}. Must be 'AG_before' or 'AG_after'")
        
        # Get fusion strategy
        self.fusion_strategy = kwargs.get('fusion_strategy', self.DEFAULT_OPTIONS['fusion_strategy'])
        
        # Initialize process group
        ranks = list(range(self.communicator.world_size))
        self.pg = dist.new_group(ranks=ranks, backend=self.backend)
        
        # Initialize fusion definition
        self.fusion = None
        self._setup_fusion()
        
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
        """Clean up process group and environment variables."""
        # Reset environment variables
        for key in self.env_vars:
            if key in os.environ:
                del os.environ[key]
        
        if hasattr(self, 'pg'):
            dist.destroy_process_group(self.pg)
    
    def _setup_fusion(self):
        """
        Set up the nvFuser fusion definition for matrix multiplication.
        This creates the fusion graph that will be used for the computation.
        """
        with FusionDefinition() as fd:
            # Define inputs
            t0 = fd.define_tensor(
                shape=[self.m, self.k],
                dtype=self.torch_dtype
            )
            t1 = fd.define_tensor(
                shape=[self.k, self.n],
                dtype=self.torch_dtype
            )
            
            # Define the matrix multiplication operation
            t2 = fd.ops.matmul(t0, t1)
            
            # Define outputs
            fd.add_output(t2)
        
        self.fusion = fd
    
    def run(self) -> torch.Tensor:
        """
        Run the TP Column-wise operation using nvFuser.
        
        Returns:
            torch.Tensor: Result matrix of shape (m, n)
        """
        # TODO: Implement the run function using nvFuser
        # This should:
        # 1. Handle the allgather operation (before or after matmul)
        # 2. Use the fusion definition for the matrix multiplication
        # 3. Handle synchronization and barriers
        pass 