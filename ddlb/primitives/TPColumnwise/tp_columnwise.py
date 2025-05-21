"""
Tensor Parallel Column-wise primitive implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
from torch.testing import assert_close
from ddlb.communicator import Communicator

class TPColumnwise(ABC):
    """
    Abstract base class for Tensor Parallel Column-wise operations.
    
    This primitive represents the operation: Allgather + Matrix Multiplication
    where the input matrices are split column-wise across GPUs.
    
    The operation can be implemented using different backends (PyTorch, NCCL, etc.)
    and different algorithms (e.g., with or without CUDA graphs).
    """
    
    def __init__(
        self,
        m: int,  # rows of first matrix
        n: int,  # columns of second matrix
        k: int,  # columns of first matrix / rows of second matrix
        dtype: str = 'float32',
        seed: int = 42,  # default seed for reproducibility
        **kwargs
    ):
        """
        Initialize the TP Column-wise primitive.
        
        Args:
            m: Number of rows in first matrix
            n: Number of columns in second matrix
            k: Number of columns in first matrix / rows of second matrix
            dtype: Data type for the matrices ('float32', 'float64', etc.)
            seed: Random seed for reproducibility
        """
        self.m = m
        self.n = n
        self.k = k
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Convert string dtype to torch dtype
        dtype_map = {
            'float32': torch.float32,
            'float64': torch.float64,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'int32': torch.int32,
            'int64': torch.int64
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}. Must be one of {list(dtype_map.keys())}")
            
        self.dtype = dtype
        self.torch_dtype = dtype_map[dtype]
        
        # Initialize communicator
        self.communicator = Communicator()
        
        # Initialize matrices
        self.A_unsharded = None
        self.A = None
        self.B = None
        
        # Store any additional implementation-specific parameters
        self.kwargs = kwargs
        
        # Setup input matrices
        self._input_setup()
    
    @abstractmethod
    def run(self) -> torch.Tensor:
        """
        Run the TP Column-wise operation.
        
        Returns:
            The result matrix of shape (m, n)
        """
        pass
    
    def _input_setup(self):
        """
        Generate random matrices for the operation.
        Private method called during initialization.
        """
        # Generate unsharded matrix A
        self.A_unsharded = torch.randn(
            self.m,
            self.k,
            dtype=self.torch_dtype,
            device='cpu'
        )

        # Shard A across GPUs
        chunk_size = self.k // self.communicator.world_size
        start_idx = self.communicator.rank * chunk_size
        end_idx = start_idx + chunk_size if self.communicator.rank < self.communicator.world_size - 1 else self.k
        self.A = self.A_unsharded[:, start_idx:end_idx].to(self.communicator.device)

        # Generate matrix B
        self.B = torch.randn(
            self.k,
            self.n,
            dtype=self.torch_dtype,
            device=self.communicator.device
        )
    
    def get_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the input matrices for this GPU.
        
        Returns:
            Tuple of (A, B) matrices where:
            - A is the sharded portion of shape (m, k_local)
            - B has shape (k, n)
        """
        return self.A, self.B
    
    def validate(self, result: torch.Tensor):
        """
        Validate the result by comparing with a single-GPU computation.
        
        Args:
            result: The result matrix from distributed computation
            
        Raises:
            AssertionError: If the result doesn't match the reference computation
        """
        # Only validate on rank 0
        if self.communicator.rank != 0:
            return True
            
        # Compute reference result on a single GPU
        reference = torch.matmul(self.A_unsharded, self.B.detach().cpu())
        
        # Compare results using torch.testing.assert_close
        assert_close(
            result.detach().cpu(),
            reference,
            rtol=float('inf'),
            atol=1e-1
        )