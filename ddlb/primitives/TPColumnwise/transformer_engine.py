"""
TransformerEngine implementation of TP Column-wise primitive
"""

import os
import torch
import torch.distributed as dist
import importlib

from .tp_columnwise import TPColumnwise
from .utils import EnvVarGuard, setup_ucc_env_vars

class TransformerEngineTPColumnwise(TPColumnwise):
    """
    TransformerEngine implementation of TP Column-wise primitive.
    
    This implementation uses NVIDIA's TransformerEngine library to optimize the matrix multiplication
    operation with FP8 precision support.
    """
    
    DEFAULT_OPTIONS = {}
    ALLOWED_VALUES = {}
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Import TransformerEngine lazily at instantiation time
        try:
            self._te = importlib.import_module('transformer_engine.pytorch')
        except Exception as e:
            raise RuntimeError(
                "TransformerEngine is required for TransformerEngineTPColumnwise but could not be imported."
            ) from e

        master_addr = os.environ.get('DDLB_MASTER_ADDR', 'localhost')
        master_port = os.environ.get('DDLB_MASTER_PORT', '12345')
        dist.init_process_group(
            backend='nccl',
            rank=self.communicator.rank,
            world_size=self.communicator.world_size,
            init_method=f"tcp://{master_addr}:{master_port}",
            device_id=self.communicator.device
        )
        self.tp_group = dist.new_group(
            ranks=list(range(self.communicator.world_size)),
            backend='nccl',
            device_id=self.communicator.device
        )

        self._te.module.base.initialize_ub(shape=[self.m, self.k],
                                tp_size=self.communicator.world_size,
                                use_fp8=False,
                                dtype=self.torch_dtype,
                                ub_cfgs=None,
                                bootstrap_backend="nccl")

        self.layer = self._te.Linear(
            in_features=self.k,
            out_features=self.n * self.communicator.world_size,
            bias=False,
            init_method=lambda weight: weight.data.copy_(self.B.t()),
            device=self.communicator.device,
            params_dtype=self.torch_dtype,
            sequence_parallel=True,
            parallel_mode='column',
            tp_group=self.tp_group,
            ub_overlap_ag=True,
            tp_size=self.communicator.world_size,
            ub_name="qkv"
        )
        self.layer.set_tensor_parallel_group(self.tp_group)

    def __del__(self):
        try:
            if hasattr(self, '_te'):
                self._te.module.base.destroy_ub()
        except Exception:
            pass
        dist.destroy_process_group()
        self.communicator.barrier()

    def run(self) -> torch.Tensor:
        return self.layer(self.A) 