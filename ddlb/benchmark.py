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
from .primitives.TPColumnwise import PyTorchTPColumnwise, ComputeOnlyTPColumnwise, FuserTPColumnwise
from .communicator import Communicator

# Configure pandas to display full output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

class PrimitiveBenchmarkRunner:
    """Main class for running distributed primitive benchmarks."""
    
    PRIMITIVES = {
        'tp_columnwise': TPColumnwise,
    }
    
    IMPLEMENTATIONS = {
        'pytorch': PyTorchTPColumnwise,
        'compute_only': ComputeOnlyTPColumnwise,
        'fuser': FuserTPColumnwise,
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
        implementation_options: Optional[Dict[str, Dict]] = None,
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
            implementation_options: Optional dictionary of implementation-specific options
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
        self.implementations = implementations
        self.implementation_options = implementation_options or {}
    
    def _create_implementation(self, impl_id: str):
        """
        Create a single implementation instance.
        
        Args:
            impl_id: Implementation identifier
            
        Returns:
            Tuple of (implementation instance, implementation options)
        """
        # Extract base implementation name and options
        if '_' in impl_id:
            base_impl = '_'.join(impl_id.split('_')[:-1])
            impl_options = self.implementation_options[impl_id]
        else:
            base_impl = impl_id
            impl_options = self.implementation_options.get(impl_id, {})
        
        if base_impl not in self.IMPLEMENTATIONS:
            raise ValueError(f"Unknown implementation: {base_impl}")
        
        # Get implementation options, merging with class defaults
        impl_class = self.IMPLEMENTATIONS[base_impl]
        options = getattr(impl_class, 'DEFAULT_OPTIONS', {}).copy()
        if impl_options:
            options.update(impl_options)
        
        # Create implementation instance
        impl = impl_class(
            m=self.m,
            n=self.n,
            k=self.k,
            dtype=self.dtype,
            **options
        )
        
        return impl, impl_options
    
    def run(self) -> pd.DataFrame:
        """
        Run the benchmarks for all implementations.
        
        Returns:
            DataFrame containing benchmark results
        """
        results = []
        comm = Communicator()
        
        # Helper lambda for tqdm iterator creation
        create_tqdm = lambda iterable, **kwargs: tqdm(iterable, **kwargs) if comm.rank == 0 else iterable
        
        # Create lists of CUDA events for timing
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_iterations)]

        for impl_id in create_tqdm(self.implementations, desc="Running benchmarks", position=0):

            if comm.rank == 0:
                print(f"Running benchmark for {impl_id} with options {self.implementation_options[impl_id]}")
            # Create implementation instance
            impl, impl_options = self._create_implementation(impl_id)
            
            try:
                # Get implementation options for result tracking
                impl_options = {k: v for k, v in impl.__dict__.items() 
                              if k in getattr(impl, 'DEFAULT_OPTIONS', {})}
                
                # Warmup runs
                for _ in create_tqdm(range(self.num_warmups), desc=f"Warming up {impl_id}", position=1, leave=False):
                    impl.run()
                
                # Actual benchmark runs
                for i in create_tqdm(range(self.num_iterations), desc=f"Running {impl_id}", position=1, leave=False):
                    start_events[i].record()
                    result = impl.run()
                    end_events[i].record()
                
                # Synchronize once after all iterations
                torch.cuda.synchronize()
                
                # Calculate elapsed times for all iterations
                times = [start_events[i].elapsed_time(end_events[i]) for i in range(self.num_iterations)]
                
                # Calculate statistics
                mean_time = np.mean(times)
                std_time = np.std(times)
                
                # Create result row with implementation options
                result_row = {
                    'implementation': impl_id,
                    'mean_time (ms)': mean_time,
                    'std_time': std_time,
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'm': self.m,
                    'n': self.n,
                    'k': self.k,
                    'dtype': self.dtype,
                    **impl_options  # Add implementation options to results
                }
                # Print result row as pandas raw output
                if comm.rank == 0: 
                    print(pd.DataFrame([result_row]).to_string(index=False))
                
                # Validate results if requested
                if self.validate:
                    try:
                        impl.validate(result)
                        result_row['valid'] = True
                    except Exception as e:
                        result_row['valid'] = False
                        print(f"Warning: Validation failed for {impl_id} with error: {e}")
                
                results.append(result_row)
            
            finally:
                # Clean up implementation
                del impl
            
            # Force garbage collection to ensure cleanup
            torch.cuda.empty_cache()
        
        # Create DataFrame and sort by mean time
        df = pd.DataFrame(results)
        
        return df
    
    def plot_results(self, results: Optional[pd.DataFrame] = None) -> None:
        """
        Plot the benchmark results.
        
        Args:
            results: Optional DataFrame of results. If None, runs the benchmarks first.
        """
        if results is None:
            results = self.run()
        
        plt.figure(figsize=(12, 6))
        
        # Create labels that include implementation options
        labels = []
        for _, row in results.iterrows():
            impl = row['implementation']
            base_impl = '_'.join(impl.split('_')[:-1]) if '_' in impl else impl
            impl_class = self.IMPLEMENTATIONS[base_impl]
            class_options = getattr(impl_class, 'DEFAULT_OPTIONS', {})
            options = {k: v for k, v in row.items() if k in class_options}
            label = f"{impl}"
            if options:
                label += f" ({', '.join(f'{k}={v}' for k, v in options.items())})"
            labels.append(label)
        
        # Plot mean times with error bars
        plt.bar(labels, results['mean_time (ms)'], 
                yerr=results['std_time'], capsize=5)
        
        plt.title(f'{self.primitive.upper()} Benchmark\n'
                 f'Size: ({self.m},{self.n},{self.k}), '
                 f'Dtype: {self.dtype}')
        plt.ylabel('Time (ms)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show() 