"""
nvFuser implementation of TP Column-wise primitive
"""

import os
import torch
import torch.distributed as dist
import nvfuser
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
            shape=[k, n], contiguity=True, dtype=torch_dtype_to_nvfuser_dtype(self.dtype)
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

class AgMatmulCollectiveBasedPipelineFusion(FusionDefinition):
    def __init__(self, dtype, m, k, n, num_devices, s, communication_backend):
        super().__init__(
            use_multidevice_executor=True, backend_type=communication_backend
        )
        self.m = m
        self.k = k
        self.n = n
        self._num_devices = num_devices
        self.dtype = dtype
        self.s = s

    def definition(self) -> None:
        m, k, n, d, s = (
            self.m,
            self.k,
            self.n,
            self._num_devices,
            self.s,
        )
        self.A = self.define_tensor(
            shape=[s, d, m // (d * s), k], contiguity=True, dtype=torch_dtype_to_nvfuser_dtype(self.dtype)
        )
        self.B = self.define_tensor(
            shape=[k, n], contiguity=True, dtype=torch_dtype_to_nvfuser_dtype(self.dtype)
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

        self.sched.parallelize(self.A, 1, ParallelType.mesh_x)
        self.sched.parallelize(self.C, 0, ParallelType.stream)


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
        'algorithm': 'default',
        's': 8  # Default pipeline size
    }
    
    ALLOWED_VALUES = {
        'backend': ['nccl', 'ucc', 'ucc/tl/nccl', 'ucc/tl/cuda'],
        'order': ['AG_before'],
        'algorithm': ['default', 'coll_pipeline'],
        's': (1, float('inf'))  # Allow any positive integer
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Parse backend configuration
        backend = self.options['backend']
        nvfuser_backend = CommunicatorBackend.nccl if backend == 'nccl' else CommunicatorBackend.ucc

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
        
        # Get algorithm and pipeline size
        self.algorithm = self.options['algorithm']
        
        nvfuser.FusionCache.reset()
        # Initialize fusion definition based on algorithm
        if self.algorithm == 'default':
            self.fusion = AgMatmulFusion(
                self.torch_dtype, 
                self.m, 
                self.k, 
                self.n, 
                self.communicator.world_size, 
                nvfuser_backend
            )
        else:  # coll_pipeline
            self.s = self.options['s']
            assert self.m % (self.communicator.world_size * self.s) == 0, "m must be divisible by s * world_size"
            self.fusion = AgMatmulCollectiveBasedPipelineFusion(
                self.torch_dtype,
                self.m,
                self.k,
                self.n,
                self.communicator.world_size,
                self.s,
                nvfuser_backend
            )
    
    def run(self) -> torch.Tensor:
        """
        Run the TP Column-wise operation using nvFuser.
        
        Returns:
            torch.Tensor: Result matrix of shape (m, n)
        """
        A, B = self.get_inputs()
        if self.algorithm == 'default':
            A = A.unsqueeze(0)
            C = self.fusion.execute([A, B])[0][0]
            C = C.reshape(C.shape[0] * C.shape[1], C.shape[2])
        else:  # coll_pipeline
            # Reshape A for pipeline algorithm
            # A is [m/d, k], we want to reshape it to [s, 1, m/(d*s), k]
            A = A.reshape(self.s, 1, A.shape[0] // self.s, A.shape[1])
            C = self.fusion.execute([A, B])[0][0]
            # C is [s, d, m/(d*s), n] and we want to first transpose the first two dimensions
            C = C.transpose(0, 1)
            # now C is [d, s, m/(d*s), n] and we reshape it to [m, n]
            C = C.reshape(self.m, self.n)
        return C
