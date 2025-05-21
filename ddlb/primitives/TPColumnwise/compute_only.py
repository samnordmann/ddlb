"""
Reference implementation that only performs the local matmul computation without the allgather.
"""

import torch
from .tp_columnwise import TPColumnwise

class ComputeOnlyTPColumnwise(TPColumnwise):
    """
    Reference implementation that only performs the local matmul computation without the allgather.
    """
    
    DEFAULT_OPTIONS = {
        'size': 'sharded',  # 'sharded' or 'unsharded'
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = self.DEFAULT_OPTIONS.copy()
        self.options.update(kwargs)
        
        # Validate size option
        assert self.options['size'] in ['sharded', 'unsharded'], \
            f"size option must be 'sharded' or 'unsharded', got {self.options['size']}"
        self.is_sharded = self.options['size'] == 'sharded'
        
        # Load A_unsharded to the device only if needed
        if not self.is_sharded:
            self.A_unsharded_gpu = self.A_unsharded.to(self.communicator.device)
    
    def run(self) -> torch.Tensor:
        """
        Run the local matmul computation using either sharded or unsharded A.
        
        Returns:
            The result matrix of shape (m, n) or (m/world_size, n) depending on size option
        """
        if self.is_sharded:
            # Compute only on local portion of A
            C = torch.matmul(self.A, self.B)
        else:
            # Compute on full matrix
            C = torch.matmul(self.A_unsharded_gpu, self.B)
        return C

    def validate(self, result: torch.Tensor) -> None:
        """
        Validate the result.
        In sharded case, skip validation since we're just doing local computation.
        In unsharded case, use base class validation.
        """
        if self.is_sharded:
            pass
        else:
            super().validate(result) 