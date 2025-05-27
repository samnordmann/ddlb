import os
import time
import torch
import torch.distributed as dist
from mpi4py import MPI

class Communicator:
    """
    A singleton class that manages distributed training environment setup and communication.
    
    This class handles:
    1. MPI/OpenMPI environment variable parsing and validation
    2. CUDA device assignment based on local rank
    3. PyTorch distributed process group initialization
    4. Master node address and port configuration
    
    The class is implemented as a singleton to ensure consistent distributed environment
    setup across the entire application. Only one instance can exist at runtime.
    
    Environment Variables:
        Required MPI variables:
            - OMPI_COMM_WORLD_RANK: Global rank of the process
            - OMPI_COMM_WORLD_LOCAL_RANK: Local rank within the node
            - OMPI_COMM_WORLD_SIZE: Total number of processes
            - OMPI_COMM_WORLD_LOCAL_SIZE: Number of processes on this node
        
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

        # Parse MPI/OpenMPI environment variables with assertions
        rank = os.environ.get('OMPI_COMM_WORLD_RANK')
        assert rank is not None, "OMPI_COMM_WORLD_RANK environment variable not set"
        self.rank = int(rank)

        local_rank = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK')
        assert local_rank is not None, "OMPI_COMM_WORLD_LOCAL_RANK environment variable not set"
        self.local_rank = int(local_rank)

        world_size = os.environ.get('OMPI_COMM_WORLD_SIZE')
        assert world_size is not None, "OMPI_COMM_WORLD_SIZE environment variable not set"
        self.world_size = int(world_size)

        local_size = os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE')
        assert local_size is not None, "OMPI_COMM_WORLD_LOCAL_SIZE environment variable not set"
        self.local_size = int(local_size)

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
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()

    def __repr__(self):
        return (
            f"Communicator(rank={self.rank}, world_size={self.world_size}, "
            f"local_rank={self.local_rank}, local_size={self.local_size}, "
            f"device={self.device}, master_addr={self.master_addr}, "
            f"master_port={self.master_port})"
        ) 