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

    Maps C = A @ B to DDLP's column-parallel linear by swapping roles:
    - A (row-sharded): weight sharded along output dimension
    - B (replicated): input activation
    All communication and compute are done inside DDLP.
    """

    DEFAULT_OPTIONS = {
        "backend": "auto",
    }

    ALLOWED_VALUES = {
        "backend": ["auto", "pytorch", "fuser", "transformer_engine"],
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

        # Map C = A @ B to DDLP: C^T = B^T @ A^T
        # DDLP input = B^T [n, k], weight = A [m/world_size, k] per rank
        self.layer = self._ddlp_primitives.LinearColumnwise(
            in_features=self.k,
            out_features=self.m,
            bias=False,
            backend=self.options["backend"],
            device=self.communicator.device,
            dtype=self.A.dtype,
        )
        with torch.no_grad():
            self.layer.weight.copy_(self.A)
        # Cache contiguous B^T to avoid implicit copy in fuser's reshape(-1, k) each forward
        self._B_t = self.B.t().contiguous()

    def __del__(self):
        if getattr(self, "_owns_process_group", False) and dist.is_initialized():
            dist.destroy_process_group()
            self.communicator.barrier()

    def run(self) -> torch.Tensor:
        # B.T: [n, k] as input; DDLP output: [n, m]; return transpose -> [m, n]
        return self.layer(self._B_t).t()
