"""
Custom kernel implementation of TP Row-wise primitive with Sequence Parallelism.

This implementation JIT compiles a CUDA kernel that handles the full TP rowwise
operation including communication. The Python side is a thin wrapper.
"""

import torch
import torch.distributed as dist
from pathlib import Path

from .tp_rowwise import TPRowwise
from ddlb.envs import get_master_addr, get_master_port


# Global cache for compiled kernel modules
_compiled_modules = {}


def _get_kernel_module(kernel_name: str):
    """
    Lazily compile and load the CUDA kernel module for the specified kernel.
    Uses JIT compilation via torch.utils.cpp_extension.load()
    """
    if kernel_name in _compiled_modules:
        return _compiled_modules[kernel_name]
    
    from torch.utils.cpp_extension import load
    
    # Get the path to the CUDA source file
    kernel_dir = Path(__file__).parent / "kernels"
    cuda_source = kernel_dir / f"{kernel_name}.cu"
    
    if not cuda_source.exists():
        raise RuntimeError(
            f"CUDA kernel source file not found: {cuda_source}\n"
            f"Available kernels: {[f.stem for f in kernel_dir.glob('*.cu')]}"
        )
    
    # Build directory for caching compiled kernels
    build_dir = kernel_dir / "build" / kernel_name
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # JIT compile the CUDA extension
    _compiled_modules[kernel_name] = load(
        name=f"tp_rowwise_{kernel_name}",
        sources=[str(cuda_source)],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-allow-unsupported-compiler",
        ],
        extra_ldflags=["-lcublas", "-lnccl"],
        build_directory=str(build_dir),
        verbose=False,
    )
    
    return _compiled_modules[kernel_name]


class CustomKernelTPRowwise(TPRowwise):
    """
    Custom CUDA kernel implementation of TP Row-wise primitive.
    
    This is a lightweight wrapper that JIT compiles a .cu file and delegates
    the entire operation (matmul + reduce-scatter) to the CUDA kernel.
    
    Options:
        kernel: Name of the .cu file to use (without extension)
    """
    
    DEFAULT_OPTIONS = {
        'kernel': 'tp_rowwise_kernel',
    }
    
    ALLOWED_VALUES = {}  # Allow any kernel name - validated at load time
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load the JIT-compiled kernel module
        kernel_name = self.options['kernel']
        self._kernel_module = _get_kernel_module(kernel_name)
        
        # Initialize torch.distributed for NCCL ID broadcast
        master_addr = get_master_addr()
        master_port = get_master_port()
        
        dist.init_process_group(
            backend='gloo',  # Use gloo for CPU tensor broadcast
            rank=self.communicator.rank,
            world_size=self.communicator.world_size,
            init_method=f"tcp://{master_addr}:{master_port}",
        )
        
        # Get NCCL unique ID from rank 0 and broadcast
        if self.communicator.rank == 0:
            nccl_id_tensor = self._kernel_module.get_nccl_unique_id()
        else:
            nccl_id_tensor = torch.zeros(128, dtype=torch.uint8)
        
        # Broadcast NCCL ID to all ranks
        dist.broadcast(nccl_id_tensor, src=0)
        
        # Initialize the kernel with communicator info and the shared NCCL ID
        self._kernel_module.init(
            self.communicator.rank,
            self.communicator.world_size,
            self.m,
            self.n,
            self.k,
            nccl_id_tensor
        )
    
    def __del__(self):
        try:
            if hasattr(self, '_kernel_module'):
                self._kernel_module.cleanup()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        try:
            self.communicator.barrier()
        except Exception:
            pass
    
    def run(self) -> torch.Tensor:
        """
        Run the TP Row-wise operation.
        
        Delegates entirely to the CUDA kernel which handles both
        the matmul and reduce-scatter communication.
        """
        return self._kernel_module.run(self.A, self.B)
