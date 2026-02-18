"""
PyTorch Symmetric Memory implementation of TP Column-wise primitive
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from .tp_columnwise import TPColumnwise
from ddlb.envs import get_master_addr, get_master_port


class PyTorchSymMemTPColumnwise(TPColumnwise):
    """
    PyTorch Symmetric Memory implementation of TP Column-wise primitive.

    This implementation uses `torch.ops.symm_mem.fused_all_gather_matmul` with
    gather_dim=0, which matches the row-sharded A layout used by TPColumnwise.
    """

    DEFAULT_OPTIONS = {}
    ALLOWED_VALUES = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        master_addr = get_master_addr()
        master_port = get_master_port()
        dist.init_process_group(
            backend="nccl",
            rank=self.communicator.rank,
            world_size=self.communicator.world_size,
            init_method=f"tcp://{master_addr}:{master_port}",
            device_id=self.communicator.device,
        )
        group_name = dist.group.WORLD.group_name

        # Some PyTorch builds still require explicit group registration for
        # symmetric memory allocations.
        enable_fn = getattr(symm_mem, "enable_symm_mem_for_group", None)
        if enable_fn is not None:
            enable_fn(group_name)

        # Allocate A in symmetric memory once and keep it for repeated runs.
        self.A_symm = symm_mem.empty(
            self.A.shape,
            dtype=self.A.dtype,
            device=self.A.device,
        )
        self.A_symm.copy_(self.A)

    def __del__(self):
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        self.communicator.barrier()

    def run(self) -> torch.Tensor:
        """
        Run TP Column-wise operation via fused all-gather + matmul.
        """
        # Keep input current in case callers mutate A between runs.
        self.A_symm.copy_(self.A)
        _, outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            self.A_symm,
            [self.B],
            gather_dim=0,
            group_name=dist.group.WORLD.group_name,
            return_A=False,
        )
        return outputs[0]
