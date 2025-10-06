"""
JAX implementation of TP Column-wise primitive
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from typing import Optional
import torch

from .tp_columnwise import TPColumnwise
from .utils import OptionsManager

class JAXTPColumnwise(TPColumnwise):
    """
    JAX implementation of TP Column-wise primitive using JAX's distributed communication.
    
    Uses JAX's mesh and sharding for distributed matmul
    """

    DEFAULT_OPTIONS = {}
    ALLOWED_VALUES = {}
    
    def __init__(self, *args, **kwargs):
        # Initialize JAX distributed before parent init
        rank = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
        world = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
        coord = os.getenv("JAX_COORD_ADDR", "127.0.0.1:12355")  # export this via mpirun

        if coord == "127.0.0.1:12355":
            print("Export JAX_COORD_ADDR=<some_host_in_job>:12355")

        jax.distributed.initialize(
            coordinator_address=coord,
            num_processes=world,
            process_id=rank,
        )

        print(f"JAX initialized with {world} processes, rank {rank}, coord {coord}")
        devices = jax.devices()

        self.mesh = jax.make_mesh(axis_shapes=(len(devices),), axis_names=('tp',))
        jax.set_mesh(self.mesh)

        self.A_sharding = NamedSharding(self.mesh, P('tp'))  # Shard along first dim
        self.B_sharding = NamedSharding(self.mesh, P(None))  # Replicated
        self.result_sharding = NamedSharding(self.mesh, P(None))  # Replicated
        
        # Call parent init which will call _input_setup
        super().__init__(*args, **kwargs)

    # Note: This will hold the torch tensors in memory as well
    # TODO: Does this change the performance at all?
    def _input_setup(self):
        super()._input_setup()
        self.A_jax = jax.numpy.asarray(self.A_unsharded, device=self.A_sharding)
        self.B_jax = jax.numpy.asarray(self.B.cpu(), device=self.B_sharding)

    def run(self) -> jnp.ndarray:
        """
        Run the TP Column-wise operation.
        
        Returns:
            jnp.ndarray: Result matrix of shape (m, n)
        """

        # The jax jit compiler will automatically parallelize the computation
        # on the tp axis based on the in and out shardings
        compute_jit = jax.jit(
            lambda A_shard, B_full: jnp.matmul(A_shard, B_full),
            in_shardings=(self.A_sharding, self.B_sharding),
            out_shardings=self.result_sharding
        )

        return compute_jit(self.A_jax, self.B_jax)

    def validate(self, result: jnp.ndarray):
        result.block_until_ready()
        result = jax.device_get(result)
        result = np.array(result) # copy because the result is not writable
        super().validate(torch.from_numpy(result))
