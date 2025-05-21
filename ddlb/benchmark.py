"""
Main benchmark runner module for distributed primitives
"""

import time
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from .primitives import TPColumnwise
from .primitives.TPColumnwise import PyTorchTPColumnwise, ComputeOnlyTPColumnwise

class PrimitiveBenchmarkRunner:
    """Main class for running distributed primitive benchmarks."""
    
    PRIMITIVES = {
        'tp_columnwise': TPColumnwise,
    }
    
    IMPLEMENTATIONS = {
        'pytorch': PyTorchTPColumnwise,
        'compute_only': ComputeOnlyTPColumnwise,
    }
    
    def __init__(
        self,
        primitive: str,
        m: int,
        n: int,
        k: int,
        implementations: List[str],
        dtype: str = 'float32',
        validate: bool = True,
        num_iterations: int = 5,
        num_warmups: int = 2,
        backend_params: Optional[Dict[str, Dict]] = None,
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            primitive: Name of the primitive to benchmark ('tp_columnwise')
            m: Number of rows in first matrix
            n: Number of columns in second matrix
            k: Number of columns in first matrix / rows in second matrix
            implementations: List of implementation names to benchmark
            dtype: Data type for the matrices
            validate: Whether to validate results
            num_iterations: Number of iterations for timing
            num_warmups: Number of warmup iterations
            backend_params: Optional dictionary of backend-specific parameters
        """
        if primitive not in self.PRIMITIVES:
            raise ValueError(f"Unknown primitive: {primitive}")
        
        self.primitive = primitive
        self.m = m
        self.n = n
        self.k = k
        self.num_iterations = num_iterations
        self.num_warmups = num_warmups
        self.dtype = dtype
        self.validate = validate
        
        # Validate implementations
        self.implementations = {}
        for impl_name in implementations:
            if impl_name not in self.IMPLEMENTATIONS:
                raise ValueError(f"Unknown implementation: {impl_name}")
            
            # Get backend-specific parameters if provided
            impl_params = backend_params.get(impl_name, {}) if backend_params else {}
            
            self.implementations[impl_name] = self.IMPLEMENTATIONS[impl_name](
                m=m,
                n=n,
                k=k,
                dtype=dtype,
                **impl_params
            )
    
    def run(self) -> pd.DataFrame:
        """
        Run the benchmarks for all implementations.
        
        Returns:
            DataFrame containing benchmark results
        """
        results = []
        
        for impl_name, impl in tqdm(self.implementations.items(), desc="Running benchmarks"):
            # Warmup runs
            for _ in range(self.num_warmups):
                impl.run()
            
            # Create CUDA events for timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Actual benchmark runs
            times = []
            for _ in range(self.num_iterations):
                start_event.record()
                result = impl.run()
                end_event.record()
                
                # Synchronize to ensure timing is accurate
                torch.cuda.synchronize()
                
                # Get elapsed time in milliseconds
                elapsed_time = start_event.elapsed_time(end_event)
                times.append(elapsed_time)
            
            # Calculate statistics
            mean_time = np.mean(times)
            std_time = np.std(times)
            
            results.append({
                'implementation': impl_name,
                'mean_time (ms)': mean_time,
                'std_time': std_time,
                'min_time': np.min(times),
                'max_time': np.max(times),
                'm': self.m,
                'n': self.n,
                'k': self.k,
                'dtype': self.dtype,
            })
            
            # Validate results if requested
            if self.validate:
                is_valid = impl.validate(result)
                results[-1]['valid'] = is_valid
                if not is_valid:
                    print(f"Warning: Validation failed for {impl_name}")
        
        return pd.DataFrame(results)
    
    def plot_results(self, results: Optional[pd.DataFrame] = None) -> None:
        """
        Plot the benchmark results.
        
        Args:
            results: Optional DataFrame of results. If None, runs the benchmarks first.
        """
        if results is None:
            results = self.run()
        
        plt.figure(figsize=(12, 6))
        
        # Plot mean times with error bars
        plt.bar(results['implementation'], results['mean_time (ms)'], 
                yerr=results['std_time'], capsize=5)
        
        plt.title(f'{self.primitive.upper()} Benchmark\n'
                 f'Size: ({self.m},{self.n},{self.k}), '
                 f'Dtype: {self.dtype}')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show() 