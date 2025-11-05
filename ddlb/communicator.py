import os
import time
import torch
import torch.distributed as dist
from .envs import get_rank, get_local_rank, get_world_size, get_local_size

class Communicator:
    """
    A singleton class that manages distributed training environment setup and communication.
    
    This class handles:
    1. MPI/OpenMPI/SLURM environment variable parsing and validation
    2. CUDA device assignment based on local rank
    3. PyTorch distributed process group initialization
    4. Master node address and port configuration
    
    The class is implemented as a singleton to ensure consistent distributed environment
    setup across the entire application. Only one instance can exist at runtime.
    
    Environment Variables:
        Supported MPI/SLURM variables (with fallback order):
            - Global rank: OMPI_COMM_WORLD_RANK → SLURM_PROCID → PMI_RANK → default "0"
            - Local rank: OMPI_COMM_WORLD_LOCAL_RANK → SLURM_LOCALID → default "0"
            - World size: OMPI_COMM_WORLD_SIZE → SLURM_NTASKS → PMI_SIZE → default "1"
            - Local size: OMPI_COMM_WORLD_LOCAL_SIZE → SLURM_NTASKS_PER_NODE → default "1"
        
        Optional DDLB variables (with defaults):
            - DDLB_MASTER_ADDR: Master node address (default: 'localhost')
            - DDLB_MASTER_PORT: Master node port (default: '12345')
    
    Example:
        >>> comm = Communicator()  # First call initializes everything
        >>> comm2 = Communicator() # Returns same instance
        >>> print(comm)  # Shows rank, world size, device info etc.
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Skip initialization if already done
        if self._initialized:
            return

        # Parse MPI/OpenMPI/SLURM environment variables with fallbacks
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.world_size = get_world_size()
        self.local_size = get_local_size()

        # Set CUDA device
        assert self.local_size <= torch.cuda.device_count(), (
            f"Local size {self.local_size} exceeds available CUDA devices {torch.cuda.device_count()}"
        )
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")

        self._initialized = True


    def barrier(self):
        """
        Synchronize all processes at this point.
        """
        # Ensure CUDA work on this device is complete
        torch.cuda.synchronize()

        # Use torch.distributed barrier if a process group is initialized
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def __repr__(self):
        return (
            f"Communicator(rank={self.rank}, world_size={self.world_size}, "
            f"local_rank={self.local_rank}, local_size={self.local_size}, "
            f"device={self.device})"
        ) 