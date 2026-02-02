"""
Tensor Parallel Row-wise primitive implementations with Sequence Parallelism
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.testing import assert_close
from ddlb.communicator import Communicator
from .utils import OptionsManager

class TPRowwise(ABC):
    """
    Abstract base class for Tensor Parallel Row-wise operations with Sequence Parallelism.
    
    This primitive represents the operation: Matrix Multiplication + Reduce-Scatter
    where the input matrices are split along the contracting dimension K across GPUs.
    After local matrix multiplication, a reduce-scatter operation combines partial results
    and shards the output along the sequence dimension (rows).
    
    Input: A[M, K_local], B[K_local, N] on each GPU
    Output: Result[M_local, N] on each GPU
    
    The operation can be implemented using different backends (PyTorch, NCCL, etc.)
    and different algorithms (e.g., with or without CUDA graphs).
    """
    
    DEFAULT_OPTIONS = {}
    ALLOWED_VALUES = {}
    
    def __init__(
        self,
        m: int,  # rows of first matrix (sequence dimension)
        n: int,  # columns of second matrix (hidden dimension)
        k: int,  # columns of first matrix / rows of second matrix (sharded dimension)
        dtype: str = 'float32',
        seed: int = 42,  # default seed for reproducibility
        **kwargs
    ):
        """
        Initialize the TP Row-wise primitive with Sequence Parallelism.
        
        Args:
            m: Number of rows in first matrix (sequence dimension, will be sharded in output)
            n: Number of columns in second matrix (hidden dimension)
            k: Number of columns in first matrix / rows of second matrix (sharded in input)
            dtype: Data type for the matrices ('float32', 'float64', etc.)
            seed: Random seed for reproducibility
        """
        # Initialize communicator
        self.communicator = Communicator()

        self.m = m
        self.n = n
        self.k = k
        # Assert that k is divisible by world_size for even sharding along contracting dimension
        if k % self.communicator.world_size != 0:
            raise ValueError(
                f"Matrix dimension k ({k}) must be divisible by world_size ({self.communicator.world_size})"
            )
        # Assert that m is divisible by world_size for even sharding of output
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
        self.B_unsharded = None
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
        Run the TP Row-wise operation with Sequence Parallelism.
        
        Performs local matmul followed by reduce-scatter to combine and shard results.
        
        Returns:
            The result matrix of shape (m_local, n) where m_local = m // world_size
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

        # Shard A across GPUs along K dimension (columns)
        chunk_size = self.k // self.communicator.world_size
        start_idx = self.communicator.rank * chunk_size
        end_idx = start_idx + chunk_size if self.communicator.rank < self.communicator.world_size - 1 else self.k
        self.A = self.A_unsharded[:, start_idx:end_idx].to(self.communicator.device)

        # Generate unsharded matrix B with uniform distribution in [-1, 1]
        self.B_unsharded = 2 * torch.rand(
            self.k,
            self.n,
            dtype=self.torch_dtype,
            device='cpu'
        ) - 1
        
        # Shard B across GPUs along K dimension (rows)
        self.B = self.B_unsharded[start_idx:end_idx, :].to(self.communicator.device)
    
    def get_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the input matrices for this GPU.
        
        Returns:
            Tuple of (A, B) matrices where:
            - A has full sequence dimension but sharded K: shape (m, k_local)
            - B is sharded along K dimension: shape (k_local, n)
        """
        return self.A, self.B
    
    def validate(self, result: torch.Tensor):
        """
        Validate the result by comparing with a single-GPU computation.
        
        Args:
            result: The sharded result matrix from distributed computation (m_local, n)
            
        Raises:
            AssertionError: If the result doesn't match the reference computation
        """
        # Compute reference result on a single GPU
        reference_full = torch.matmul(self.A_unsharded, self.B_unsharded)
        
        # Extract the portion that should match this GPU's output
        chunk_size = self.m // self.communicator.world_size
        start_idx = self.communicator.rank * chunk_size
        end_idx = start_idx + chunk_size if self.communicator.rank < self.communicator.world_size - 1 else self.m
        reference = reference_full[start_idx:end_idx, :]

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

