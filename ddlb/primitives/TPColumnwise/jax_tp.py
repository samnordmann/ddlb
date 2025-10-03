"""
JAX implementation of TP Column-wise primitive
"""

import os
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from typing import Optional

from .tp_columnwise import TPColumnwise
from .utils import OptionsManager

class JAXTPColumnwise(TPColumnwise):
    """
    JAX implementation of TP Column-wise primitive using JAX's distributed communication.
    Performs Allgather on A followed by matrix multiplication with B.
    
    Uses JAX's mesh and sharding for distributed computation across multiple devices.
    
    The order of operations can be controlled using the 'order' option:
    - 'AG_before': First perform allgather, then matmul (default)
    - 'AG_after': First perform local matmul, then allgather results
    """
    
    DEFAULT_OPTIONS = {
        'order': 'AG_before'  # Default order
    }
    
    ALLOWED_VALUES = {
        'order': ['AG_before', 'AG_after']
    }
    
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
        
        # Create device mesh for distributed computation
        # Assumes devices are arranged in a single row for tensor parallelism
        devices = jax.devices()

        self.mesh = Mesh(devices, axis_names=('tp',))
        
        # Set up sharding specs before parent init calls _input_setup
        self.A_sharding = NamedSharding(self.mesh, P('tp', None))  # Shard along first dim
        self.B_sharding = NamedSharding(self.mesh, P(None, None))  # Replicated
        self.result_sharding = NamedSharding(self.mesh, P(None, None))  # Replicated
        
        # Now call parent init which will call _input_setup
        super().__init__(*args, **kwargs)
        
        # Get allgather order after options are parsed
        self.order = self.options['order']

    def __del__(self):
        # JAX distributed cleanup happens automatically
        pass

    def _input_setup(self):
        """
        Generate random matrices for the operation using JAX arrays.
        Private method called during initialization.
        """
        # Map string dtypes to JAX dtypes
        dtype_map = {
            'float32': jnp.float32,
            'float64': jnp.float64,
            'float16': jnp.float16,
            'bfloat16': jnp.bfloat16,
            'int32': jnp.int32,
            'int64': jnp.int64
        }
        jax_dtype = dtype_map.get(self.dtype, jnp.float32)
        
        # Set JAX random key for reproducibility
        key = jax.random.PRNGKey(42)  # Use same seed as torch version
        key_A, key_B = jax.random.split(key)
        
        # Generate unsharded matrix A with uniform distribution in [-1, 1]
        self.A_unsharded = jax.random.uniform(
            key_A,
            shape=(self.m, self.k),
            dtype=jax_dtype,
            minval=-1, 
            maxval=1
        )

        # Shard A across devices - convert to sharded array
        self.A = jax.device_put(self.A_unsharded, self.A_sharding)

        # Generate matrix B with uniform distribution in [-1, 1]
        B_full = jax.random.uniform(
            key_B,
            shape=(self.k, self.n),
            dtype=jax_dtype,
            minval=-1,
            maxval=1
        )
        
        # Replicate B across all devices
        self.B = jax.device_put(B_full, self.B_sharding)

    def run(self) -> jnp.ndarray:
        """
        Run the TP Column-wise operation.
        
        Returns:
            jnp.ndarray: Result matrix of shape (m, n)
        """
      def _compute_ag_before(A_shard, B_full):
          """Allgather first, then matmul"""
          A_gathered = jax.lax.all_gather(A_shard, axis_name='tp', axis=0, tiled=True)
          return jnp.matmul(A_gathered, B_full)

       def _compute_ag_after(A_shard, B_full):
            """Matmul first, then allgather results"""
            local_result = jnp.matmul(A_shard, B_full)
            return jax.lax.all_gather(local_result, axis_name='tp', axis=0, tiled=True)
        
        sharded_fn = shard_map(
            _compute_ag_before if self.order == 'AG_before' else _compute_ag_after,
            mesh=self.mesh,
            in_specs=(P('tp', None), P(None, None)),  # A sharded, B replicated
            out_specs=P(None, None),  # Result replicated
            check_rep=False
        )
        return sharded_fn(self.A, self.B)
            

    def get_inputs(self):
        """
        Get the input matrices for this GPU.
        
        Returns:
            Tuple of (A, B) JAX arrays where:
            - A is the sharded portion
            - B is replicated 
        """
        return self.A, self.B

    def validate(self, result: jnp.ndarray):
        """
        Validate the result by comparing with a single-device computation.
        
        Args:
            result: The result matrix from distributed computation
            
        Raises:
            AssertionError: If the result doesn't match the reference computation
        """
        # Compute reference result
        reference = jnp.matmul(self.A_unsharded, self.B)

        # Set tolerance based on dtype
        if result.dtype in (jnp.float16, jnp.bfloat16):
            atol = 1e-3
        else:
            atol = 1e-4
        atol *= self.k  # for accumulated error

        # Compare results using JAX's allclose
        assert jnp.allclose(result, reference, rtol=0, atol=atol), \
            f"Result validation failed. Max difference: {jnp.max(jnp.abs(result - reference))}"
