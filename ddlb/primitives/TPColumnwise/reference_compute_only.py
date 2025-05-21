"""
Reference implementation that only performs the local matmul computation without the allgather.
"""

import torch
from .tp_columnwise import TPColumnwise

class ReferenceComputeOnly(TPColumnwise):
    """
    Reference implementation that only performs the local matmul computation without the allgather.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load A_unsharded to the device
        self.A_unsharded_gpu = self.A_unsharded.to(self.communicator.device)
    
    def run(self) -> torch.Tensor:
        """
        Run the local matmul computation using A_unsharded.
        
        Returns:
            The result matrix of shape (m, n)
        """
        # Compute local matmul using A_unsharded
        C = torch.matmul(self.A_unsharded_gpu, self.B)
        return C 