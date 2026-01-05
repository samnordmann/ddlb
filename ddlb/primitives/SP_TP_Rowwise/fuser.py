"""
nvFuser implementation of SP_TP_Rowwise primitive
"""

import os
import torch
import torch.distributed as dist
import nvfuser_direct as nvfuser
from nvfuser_direct import DataType, FusionDefinition, CommunicatorBackend, ParallelType
from nvfuser_direct.pytorch_utils import torch_dtype_to_nvfuser_dtype

from .sp_tp_rowwise import SP_TP_Rowwise
from .utils import EnvVarGuard, setup_ucc_env_vars


class AgMatmulFusion(FusionDefinition):
    def __init__(self, dtype, m, k, n, num_devices):
        super().__init__()
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
        mesh = nvfuser.multidevice.DeviceMesh(range(self._num_devices))
        for tv in [
            self.A,
            self.B,
            self.C,
        ]:
            tv.set_device_mesh(mesh)

        self.A.axis(0).parallelize(ParallelType.mesh_x)

class AgMatmulCollectiveBasedPipelineFusion(FusionDefinition):
    def __init__(self, dtype, m, k, n, num_devices, s, communication_backend):
        super().__init__()
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
        mesh = nvfuser.multidevice.DeviceMesh(range(self._num_devices))
        for tv in [
            self.A,
            self.B,
            self.C,
        ]:
            tv.set_device_mesh(mesh)

        self.A.axis(1).parallelize(ParallelType.mesh_x)
        self.C.axis(0).parallelize(ParallelType.stream)

class AgMatmulP2PBasedPipelineFusion(FusionDefinition):
    def __init__(self, dtype, m, k, n, num_devices, communication_backend):
        super().__init__()
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
            self._num_devices
        )
        self.A = self.define_tensor(
            shape=[d, m // d , k], contiguity=True, dtype=torch_dtype_to_nvfuser_dtype(self.dtype)
        )
        self.B = self.define_tensor(
            shape=[k, n], contiguity=True, dtype=torch_dtype_to_nvfuser_dtype(self.dtype)
        )

        self.C = self.ops.matmul(
            self.A, self.B
        ) 

        self.add_output(self.C)

    def multidevice_schedule(self):
        mesh = nvfuser.multidevice.DeviceMesh(range(self._num_devices))
        for tv in [
            self.A,
            self.B,
            self.C,
        ]:
            tv.set_device_mesh(mesh)

        self.A.axis(0).parallelize(ParallelType.mesh_x)
        self.C.axis(0).parallelize(ParallelType.stream)


class Fuser_SP_TP_Rowwise(SP_TP_Rowwise):
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
        'backend': ['nccl', 'ucc', 'ucc/tl/nccl', 'ucc/tl/cuda', 'cuda'],
        'order': ['AG_before', 'AG_after'],
        'algorithm': ['default', 'coll_pipeline', 'p2p_pipeline'],
        's': (1, float('inf'))  # Allow any positive integer
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Parse backend configuration
        backend = self.options['backend']
        nvfuser_backend = CommunicatorBackend.nccl if backend == 'nccl' else CommunicatorBackend.cuda if backend == 'cuda' else CommunicatorBackend.ucc

        if backend.startswith('ucc/tl/'):
            self.tl = backend.split('/')[-1]
            self.backend = 'ucc'
        else:
            self.backend = backend
            self.tl = None
        
        # Get allgather order
        self.order = self.options['order']

        # Set up environment variables (include nvFuser flags)
        _env_vars = setup_ucc_env_vars(backend)
        enable_flags = []
        if self.order == 'AG_after':
            enable_flags.append('insert_resharding_after')
        if enable_flags:
            _env_vars['NVFUSER_ENABLE'] = ','.join(enable_flags)
        self.env_guard = EnvVarGuard(_env_vars)
        
        # Get algorithm and pipeline size
        self.algorithm = self.options['algorithm']
        
        # Initialize fusion definition based on algorithm
        if self.algorithm == 'default':
            self.fusion = AgMatmulFusion(
                self.torch_dtype, 
                self.m, 
                self.k, 
                self.n, 
                self.communicator.world_size
            )
        elif self.algorithm == 'coll_pipeline':  # coll_pipeline
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
        elif self.algorithm == 'p2p_pipeline':  # p2p_pipeline
            self.fusion = AgMatmulP2PBasedPipelineFusion(
                self.torch_dtype,
                self.m,
                self.k,
                self.n,
                self.communicator.world_size,
                nvfuser_backend
            )
        params = nvfuser.multidevice.MultiDeviceExecutorParams()
        params.backend_type = nvfuser_backend
        params.use_allocation_cache = True
        with self.fusion:
            self.fusion.definition()
            self.fusion.multidevice_schedule()
        self.multidevice_executor = nvfuser.multidevice.MultiDeviceExecutor(
            self.fusion.fusion, params
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
            # C = self.fusion.execute([A, B])[0][0]
            C = self.multidevice_executor.run([A, B])[0]
            C = C.reshape(C.shape[0] * C.shape[1], C.shape[2])
        elif self.algorithm == 'coll_pipeline':  # coll_pipeline
            # Reshape A for pipeline algorithm
            # A is [m/d, k], we want to reshape it to [s, 1, m/(d*s), k]
            A = A.reshape(self.s, 1, A.shape[0] // self.s, A.shape[1])
            C = self.multidevice_executor.run([A, B])[0]
            # C is [s, d, m/(d*s), n] and we want to first transpose the first two dimensions
            C = C.transpose(0, 1)
            # now C is [d, s, m/(d*s), n] and we reshape it to [m, n]
            C = C.reshape(self.m, self.n)
        elif self.algorithm == 'p2p_pipeline':  # p2p_pipeline
            A = A.unsqueeze(0)
            C = self.multidevice_executor.run([A, B])[0]
            C = C.reshape(C.shape[0] * C.shape[1], C.shape[2])
        return C
