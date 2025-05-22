"""
Tensor Parallel Column-wise primitive implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.testing import assert_close
from ddlb.communicator import Communicator
from .utils import OptionsManager

class TPColumnwise(ABC):
    """
    Abstract base class for Tensor Parallel Column-wise operations.
    
    This primitive represents the operation: Allgather + Matrix Multiplication
    where the input matrices are split column-wise across GPUs.
    
    The operation can be implemented using different backends (PyTorch, NCCL, etc.)
    and different algorithms (e.g., with or without CUDA graphs).
    """
    
    DEFAULT_OPTIONS = {}
    ALLOWED_VALUES = {}
    
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
        # Initialize communicator
        self.communicator = Communicator()

        self.m = m
        self.n = n
        self.k = k
        # Assert that m is divisible by world_size for even sharding
        if m % self.communicator.world_size != 0:
            raise ValueError(
                f"Matrix dimension m ({m}) must be divisible by world_size ({self.communicator.world_size})"
            )
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
        
        # Initialize matrices
        self.A_unsharded = None
        self.A = None
        self.B = None
        
        # Initialize options manager with class defaults
        self.options = OptionsManager(self.DEFAULT_OPTIONS, self.ALLOWED_VALUES)
        self.options.parse(kwargs)
        
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
        # Generate unsharded matrix A with uniform distribution in [-1, 1]
        self.A_unsharded = 2 * torch.rand(
            self.m,
            self.k,
            dtype=self.torch_dtype,
            device='cpu'
        ) - 1

        # Shard A across GPUs
        chunk_size = self.m // self.communicator.world_size
        start_idx = self.communicator.rank * chunk_size
        end_idx = start_idx + chunk_size if self.communicator.rank < self.communicator.world_size - 1 else self.m
        self.A = self.A_unsharded[start_idx:end_idx, :].to(self.communicator.device)

        # Generate matrix B with uniform distribution in [-1, 1]
        self.B = 2 * torch.rand(
            self.k,
            self.n,
            dtype=self.torch_dtype,
            device=self.communicator.device
        ) - 1
    
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
        # Compute reference result on a single GPU
        reference = torch.matmul(self.A_unsharded, self.B.detach().cpu())

        if self.torch_dtype in (torch.float16, torch.bfloat16):
            atol = 1e-3
        else:
            atol = 1e-4
        atol *= (self.k) # for accumulated error

        # Compare results using torch.testing.assert_close
        assert_close(
            result.detach().cpu(),
            reference,
            rtol=0,
            atol=atol
        )