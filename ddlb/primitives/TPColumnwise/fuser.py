"""
nvFuser implementation of TP Column-wise primitive
"""

import os
import torch
import torch.distributed as dist
from nvfuser import DataType, FusionDefinition, CommunicatorBackend, DeviceMesh, ParallelType
from nvfuser.pytorch_utils import torch_dtype_to_nvfuser_dtype

from .tp_columnwise import TPColumnwise
from .utils import EnvVarGuard, setup_ucc_env_vars


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
    
    ALLOWED_VALUES = {
        'backend': ['nccl', 'ucc', 'ucc/tl/nccl', 'ucc/tl/cuda'],
        'order': ['AG_before']
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
        
        # Set up environment variables
        self.env_guard = EnvVarGuard(setup_ucc_env_vars(backend))
    
        # Get allgather order
        self.order = self.options['order']
        
        # Initialize fusion definition
        self.fusion = AgMatmulFusion(self.torch_dtype, self.m, self.k, self.n, self.communicator.world_size, CommunicatorBackend.nccl)
    
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
