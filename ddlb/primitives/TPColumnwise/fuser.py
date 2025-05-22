"""
nvFuser implementation of TP Column-wise primitive
"""

import os
import torch
import torch.distributed as dist
from nvfuser import DataType, FusionDefinition, CommunicatorBackend, DeviceMesh, ParallelType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype

from .tp_columnwise import TPColumnwise


class AgMatmulFusion(FusionDefinition):
    def __init__(self, dtype, m, k, n, num_devices, communication_backend):
        super().__init__(
            use_multidevice_executor=True, backend_type=communication_backend
        )
        self.m = m
        self.k = k
        self.n = n
        self._num_devices = num_devices
        self.dtype = dtype

    def definition(self) -> None:
        m, k, n, d = (
            self.m,
            self.k,
            self.n,
            self._num_devices,
        )
        self.A = self.define_tensor(
            shape=[d, m // d, k], contiguity=True, dtype=torch_dtype_to_nvfuser_dtype(self.dtype)
        )
        self.B = self.define_tensor(
            shape=[n, k], contiguity=True, dtype=torch_dtype_to_nvfuser_dtype(self.dtype)
        )

        self.C = self.ops.matmul(
            self.A, self.B
        ) 

        self.add_output(self.C)

    def multidevice_schedule(self):
        mesh = DeviceMesh(range(self._num_devices))
        for tv in [
            self.A,
            self.B,
            self.C,
        ]:
            self.sched._set_device_mesh(tv, mesh)

        self.sched.parallelize(self.A, 0, ParallelType.mesh_x)


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
        if self.order not in ['AG_before']:
            raise ValueError(f"Invalid order: {self.order}. Must be 'AG_before' or 'AG_after'")
        
        # # Get fusion strategy
        # self.fusion_strategy = kwargs.get('fusion_strategy', self.DEFAULT_OPTIONS['fusion_strategy'])
        
        
        # Initialize fusion definition
        self.fusion = AgMatmulFusion(self.torch_dtype, self.m, self.k, self.n, self.communicator.world_size, CommunicatorBackend.nccl)
    
    def __del__(self):
        """Clean up process group and environment variables."""
        # Reset environment variables
        for key in self.env_vars:
            if key in os.environ:
                del os.environ[key]
    
    def run(self) -> torch.Tensor:
        """
        Run the TP Column-wise operation using nvFuser.
        
        Returns:
            torch.Tensor: Result matrix of shape (m, n)
        """
        A, B = self.get_inputs()
        A = A.unsqueeze(0)
        C = self.fusion.execute([A, B])[0][0]
        C = C.reshape(C.shape[0] * C.shape[1], C.shape[2])
        return C
