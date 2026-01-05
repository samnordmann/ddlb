"""
Reference implementation that only performs the local matmul computation without the allgather.
"""

import torch
from .sp_tp_rowwise import SP_TP_Rowwise

class ComputeOnly_SP_TP_Rowwise(SP_TP_Rowwise):
    """
    Reference implementation that only performs the local matmul computation without the allgather.
    """
    
    DEFAULT_OPTIONS = {
        'size': 'sharded',  # 'sharded' or 'unsharded'
    }
    
    ALLOWED_VALUES = {
        'size': ['sharded', 'unsharded']
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Get size option
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