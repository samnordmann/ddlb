"""
DDLP implementation of TP Column-wise primitive.
"""

import importlib

import torch
import torch.distributed as dist

from ddlb.envs import get_master_addr, get_master_port
from .tp_columnwise import TPColumnwise


class DDLPTPColumnwise(TPColumnwise):
    """
    TP Column-wise implementation powered by the optional DDLP package.

    DDLP's LinearColumnwise accepts row-sharded input A [m/ws, k] and
    replicated weight B [k, n], matching fused_all_gather_matmul semantics.
    """

    DEFAULT_OPTIONS = {
        "backend": "auto",
    }

    ALLOWED_VALUES = {
        "backend": ["auto", "pytorch", "fuser"],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            self._ddlp_primitives = importlib.import_module("ddlp.primitives")
        except Exception as exc:
            raise RuntimeError(
                "DDLP is required for DDLPTPColumnwise but could not be imported. "
                "Install DDLP to use implementation='ddlp'."
            ) from exc

        self._owns_process_group = False
        if not dist.is_initialized():
            master_addr = get_master_addr()
            master_port = get_master_port()
            dist.init_process_group(
                backend="nccl",
                rank=self.communicator.rank,
                world_size=self.communicator.world_size,
                init_method=f"tcp://{master_addr}:{master_port}",
                device_id=self.communicator.device,
            )
            self._owns_process_group = True

        self.layer = self._ddlp_primitives.LinearColumnwise(
            in_features=self.k,
            out_features=self.n,
            bias=False,
            backend=self.options["backend"],
            device=self.communicator.device,
            dtype=self.A.dtype,
        )
        with torch.no_grad():
            self.layer.weight.copy_(self.B)

    def __del__(self):
        if getattr(self, "_owns_process_group", False) and dist.is_initialized():
            dist.destroy_process_group()
            self.communicator.barrier()

    def run(self) -> torch.Tensor:
        return self.layer(self.A)
