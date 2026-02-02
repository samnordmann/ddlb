"""
nvFuser implementation of TP Row-wise primitive with Sequence Parallelism
"""

import os
import torch
import torch.distributed as dist
import nvfuser_direct as nvfuser
from nvfuser_direct import DataType, FusionDefinition, CommunicatorBackend, ParallelType
from nvfuser_direct.pytorch_utils import torch_dtype_to_nvfuser_dtype

from .tp_rowwise import TPRowwise
from .utils import EnvVarGuard, setup_ucc_env_vars

class MatmulRsFusion(FusionDefinition):
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
            shape=[d, m, k // d], contiguity=True, dtype=torch_dtype_to_nvfuser_dtype(self.dtype) # [didx(d), m, k/d]
        )
        self.B = self.define_tensor(
            shape=[d, k // d, n], contiguity=True, dtype=torch_dtype_to_nvfuser_dtype(self.dtype) # [didx(d), k/d, n]
        )

        self.C_mm = self.ops.matmul(
            self.A, self.B # [didx(d), m, n]
        )

        self.C_reshaped = self.ops.reshape(self.C_mm, [d, d, m // d, n]) # [didx(d), d, m/d, n]

        self.C = self.ops.sum(self.C_reshaped, 0) # [r(d), didx(d), m/d, n]

        self.add_output(self.C)

    def multidevice_schedule(self):
        mesh = nvfuser.multidevice.DeviceMesh(range(self._num_devices))
        for tv in [
            self.A,
            self.B,
            self.C_mm,
            self.C_reshaped,
        ]:
            tv.set_device_mesh(mesh)
            tv.axis(0).parallelize(ParallelType.mesh_x)
        
        self.C.set_device_mesh(mesh)
        self.C.axis(1).parallelize(ParallelType.mesh_x)

class MatmulRsCollectiveBasedPipelineFusion(FusionDefinition):
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
            shape=[d, s, d, m // (s*d), k // d], contiguity=True, dtype=torch_dtype_to_nvfuser_dtype(self.dtype) # [didx(d), s, d, m/(s*d), k/d]
        )
        self.B = self.define_tensor(
            shape=[1, k // d, n], contiguity=True, dtype=torch_dtype_to_nvfuser_dtype(self.dtype) # [didx(d), k/d, n]
        )

        self.broadcast_B = self.ops.broadcast(self.B, [False, True, True, False, False]) # [didx(d), s, 1, k/d, n]

        self.C_unreduced = self.ops.matmul(
            self.A, self.broadcast_B # [didx(d), stream(s), d, m/(s*d), n]
        )

        self.C = self.ops.sum(self.C_unreduced, 0) # [r(d), stream(s), didx(d), m/(s*d), n]

        self.add_output(self.C)

    def multidevice_schedule(self):
        mesh = nvfuser.multidevice.DeviceMesh(range(self._num_devices))
        for tv in [
            self.A,
            self.B,
            self.broadcast_B,
            self.C_unreduced,
            self.C,
        ]:
            tv.set_device_mesh(mesh)

        self.A.axis(0).parallelize(ParallelType.mesh_x)
        self.B.axis(0).parallelize(ParallelType.mesh_x)
        self.broadcast_B.axis(0).parallelize(ParallelType.mesh_x)
        self.C_unreduced.axis(1).parallelize(ParallelType.stream)
        self.C_unreduced.axis(0).parallelize(ParallelType.mesh_x)
        self.C.axis(1).parallelize(ParallelType.stream)
        self.C.axis(2).parallelize(ParallelType.mesh_x)

class MatmulRsP2PBasedPipelineFusion(FusionDefinition):
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
            self._num_devices,
        )
        self.A = self.define_tensor(
            shape=[d, d, m // d, k // d], contiguity=True, dtype=torch_dtype_to_nvfuser_dtype(self.dtype) # [didx(d), stream(d), m/d, k/d]
        )
        self.B = self.define_tensor(
            shape=[d, 1, k // d, n], contiguity=True, dtype=torch_dtype_to_nvfuser_dtype(self.dtype) # [didx(d), k/d, n]
        )

        self.C_unreduced = self.ops.matmul(
            self.A, self.B # [stream(d), didx(d), m/d, n]
        )

        self.C = self.ops.sum(self.C_unreduced, 0) # [r(d), didx(d), m/d, n]

        self.add_output(self.C)

    def multidevice_schedule(self):
        mesh = nvfuser.multidevice.DeviceMesh(range(self._num_devices))
        for tv in [
            self.A,
            self.B,
            self.C_unreduced,
            self.C,
        ]:
            tv.set_device_mesh(mesh)

        # Only the reduced dimension (d) is actually sharded, M is split
        # logically for convenience
        self.A.axis(0).parallelize(ParallelType.mesh_x)
        self.A.axis(1).parallelize(ParallelType.stream)
        self.B.axis(0).parallelize(ParallelType.mesh_x)
        self.C_unreduced.axis(1).parallelize(ParallelType.mesh_x)
        self.C_unreduced.axis(0).parallelize(ParallelType.stream)
        self.C.axis(1).parallelize(ParallelType.mesh_x)

class FuserTPRowwise(TPRowwise):
    """
    nvFuser implementation of TP Row-wise primitive with Sequence Parallelism.
    
    This implementation uses NVIDIA's nvFuser library to optimize the matrix multiplication
    and reduce-scatter operations. The fusion is done at the CUDA kernel level, which can 
    provide better performance than standard PyTorch operations.
    
    The implementation supports both NCCL and UCC backends, similar to the PyTorch implementation.
    """
    
    DEFAULT_OPTIONS = {
        'backend': 'nccl',  # Default backend
        'algorithm': 'default',
        's': 8  # Default pipeline size
    }
    
    ALLOWED_VALUES = {
        'backend': ['nccl', 'ucc', 'ucc/tl/nccl', 'ucc/tl/cuda', 'cuda'],
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
        
        # Set up environment variables (include nvFuser flags)
        _env_vars = setup_ucc_env_vars(backend)
        # Enable reduce-scatter resharding
        enable_flags = ['insert_resharding_after']
        if enable_flags:
            _env_vars['NVFUSER_ENABLE'] = ','.join(enable_flags)
        self.env_guard = EnvVarGuard(_env_vars)
        
        # Get algorithm and pipeline size
        self.algorithm = self.options['algorithm']
        self.s = self.options['s']
        
        # Initialize fusion definition based on algorithm
        if self.algorithm == 'default':
            self.fusion = MatmulRsFusion(
                self.torch_dtype, 
                self.m, 
                self.k, 
                self.n, 
                self.communicator.world_size
            )
        elif self.algorithm == 'coll_pipeline':  # coll_pipeline
            assert self.m % (self.communicator.world_size * self.s) == 0, "m must be divisible by s * world_size for the coll_pipeline algorithm"
            assert self.k % self.communicator.world_size == 0, "k must be divisible by world_size"
            self.fusion = MatmulRsCollectiveBasedPipelineFusion(
                self.torch_dtype,
                self.m,
                self.k,
                self.n,
                self.communicator.world_size,
                self.s,
                nvfuser_backend
            )
        elif self.algorithm == 'p2p_pipeline':  # p2p_pipeline
            assert self.m % self.communicator.world_size == 0, "m must be divisible by world_size"
            assert self.k % self.communicator.world_size == 0, "k must be divisible by world_size"
            if self.s != self.communicator.world_size:
                print(f"Warning: s={self.s} is not equal to world_size={self.communicator.world_size} for p2p_pipeline algorithm. Ignoring s and using world_size instead.")
                self.s = self.communicator.world_size
            self.fusion = MatmulRsP2PBasedPipelineFusion(
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
        params.number_of_streams = self.s
        with self.fusion:
            self.fusion.definition()
            self.fusion.multidevice_schedule()
        self.multidevice_executor = nvfuser.multidevice.MultiDeviceExecutor(
            self.fusion.fusion, params
        )

    def run(self) -> torch.Tensor:
        """
        Run the TP Row-wise operation using nvFuser with Sequence Parallelism.
        
        Performs local matmul followed by reduce-scatter.
        
        Returns:
            torch.Tensor: Result matrix of shape (m_local, n) where m_local = m // world_size
        """
        A, B = self.get_inputs()
        if self.algorithm == 'default':
            A = A.unsqueeze(0)
            B = B.unsqueeze(0)
            C = self.multidevice_executor.run([A, B])[0]
            C = C.squeeze(0)
        elif self.algorithm == 'coll_pipeline':  # coll_pipeline
            A = A.view(1, self.s, self.communicator.world_size, self.m//(self.s*self.communicator.world_size), self.k//self.communicator.world_size) # [didx(d), m, k//d] -> [1, s, d, m/(s*d), k/d]
            B = B.unsqueeze(0)
            C = self.multidevice_executor.run([A, B])[0] # [r(d), didx(d), m/d, n]
            C = C.view(self.m // self.communicator.world_size, self.n)
        elif self.algorithm == 'p2p_pipeline':  # p2p_pipeline
            # A is [didx(d), d, m//d, k//d], B is [didx(d), k//d, n]
            # C will be [r(d), didx(d), m/d, n] after reduce-scatter
            A = A.unsqueeze(0)
            A = A.view(1, self.communicator.world_size, self.m//self.communicator.world_size, self.k//self.communicator.world_size)
            B = B.unsqueeze(0)
            B = B.unsqueeze(0) # [1, 1, k/d, n]
            C = self.multidevice_executor.run([A, B])[0]
            C = C.view(self.m // self.communicator.world_size, self.n)
        return C

