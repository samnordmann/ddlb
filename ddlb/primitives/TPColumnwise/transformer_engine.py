"""
TransformerEngine implementation of TP Column-wise primitive
"""

import os
import torch
import torch.distributed as dist
# import transformer_engine as te
# from transformer_engine.pytorch import fp8_autocast
import transformer_engine.pytorch as te


from .tp_columnwise import TPColumnwise
from .utils import EnvVarGuard, setup_ucc_env_vars

class TransformerEngineTPColumnwise(TPColumnwise):
    """
    TransformerEngine implementation of TP Column-wise primitive.
    
    This implementation uses NVIDIA's TransformerEngine library to optimize the matrix multiplication
    operation with FP8 precision support. The implementation supports both NCCL and UCC backends.
    
    The operation can be performed in two orders:
    - 'AG_before': First perform allgather, then matmul (default)
    - 'AG_after': First perform local matmul, then allgather results
    """
    
    DEFAULT_OPTIONS = {'backend': 'nccl'}
    
    ALLOWED_VALUES = {'backend': ['nccl']}
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        te.module.base.initialize_ub(shape=[self.n, self.k],
                                tp_size=self.communicator.world_size,
                                use_fp8=False,
                                dtype=self.torch_dtype,
                                ub_cfgs=None,
                                bootstrap_backend=None)

        def init_weights(weight):
            # B is of shape (k, n) and we need to transpose it to (n, k) for the linear layer
            weight.data.copy_(self.A.to(self.communicator.device))
            return weight

        self.layer = te.Linear(
            in_features=self.k,
            out_features=self.m,
            bias=False,
            init_method=init_weights,
            device=self.communicator.device,
            params_dtype = self.torch_dtype,
            # --- Key Parameters ---
            sequence_parallel=True,
            parallel_mode='column', # Set for TP-Columnwise
            tp_group=self.tp_group,
            # --------------------
            ub_overlap_ag=True,
            tp_size=self.communicator.world_size,
            ub_name="qkv"
        )

        self.layer.set_tensor_parallel_group(self.tp_group)




        # Init ub https://github.com/NVIDIA/TransformerEngine/blob/cd37379d24f574f98e74e81d242c1419f1159e6e/transformer_engine/pytorch/module/base.py#L109

    def __del__(self):
        dist.destroy_process_group()
        self.communicator.barrier()


    def run(self) -> torch.Tensor:
        """
        Run the TP Column-wise operation using TransformerEngine.
        
        Returns:
            torch.Tensor: Result matrix of shape (m, n)
        """
        # Implementation: https://github.com/NVIDIA/TransformerEngine/blob/cd37379d24f574f98e74e81d242c1419f1159e6e/transformer_engine/pytorch/csrc/extensions/gemm.cpp#L89
        # https://github.com/NVIDIA/TransformerEngine/blob/cd37379d24f574f98e74e81d242c1419f1159e6e/transformer_engine/pytorch/cpp_extensions/gemm.py#L24
        # https://github.com/NVIDIA/TransformerEngine/blob/v1.9/transformer_engine/pytorch/csrc/comm_gemm_overlap.h
        # https://github.com/NVIDIA/TransformerEngine/blob/cd37379d24f574f98e74e81d242c1419f1159e6e/transformer_engine/pytorch/module/linear.py#L894
        # https://github.com/NVIDIA/TransformerEngine/blob/cd37379d24f574f98e74e81d242c1419f1159e6e/transformer_engine/pytorch/module/linear.py#L1043



# File "/usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/module/linear.py", line 129, in forward
# assert inp_shape[-1] == in_features, "GEMM not possible"
        result = self.layer(self.B.t())





        # te.generic_gemm(
        #                                         in_features=1024,
        #                                         out_features=1024,
        #                                         bias=False,
        #                                         # Key parameter: Set to 'column' for TP-Columnwise
        #                                         parallel_mode='column',
        #                                         # # Provide the tensor parallel group (can also be set later)
        #                                         # tp_group=tp_group
        #                                     )

        # result = torch.matmul(self.A, self.B)

        return result 