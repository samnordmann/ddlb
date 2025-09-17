"""
Main benchmark runner module for distributed primitives
"""

import os
import time
import multiprocessing as mp
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Avoid importing CUDA-dependent primitives in the parent process.

def _benchmark_worker_entry(
    impl_id: str,
    m: int,
    n: int,
    k: int,
    dtype: str,
    num_warmups: int,
    num_iterations: int,
    impl_opts: Dict,
    validate: bool,
    result_queue,
):
    # Import CUDA-dependent libraries inside child process only
    import torch
    import numpy as _np
    from ddlb.primitives.TPColumnwise import (
        PyTorchTPColumnwise as _PyTorch,
        ComputeOnlyTPColumnwise as _ComputeOnly,
        FuserTPColumnwise as _Fuser,
        TransformerEngineTPColumnwise as _TE,
    )

    IMPLEMENTATIONS_LOCAL = {
        'pytorch': _PyTorch,
        'compute_only': _ComputeOnly,
        'fuser': _Fuser,
        'transformer_engine': _TE,
    }

    # Parse base implementation and options
    if '_' in impl_id:
        base_impl = '_'.join(impl_id.split('_')[:-1])
    else:
        base_impl = impl_id

    impl_class = IMPLEMENTATIONS_LOCAL[base_impl]
    options = getattr(impl_class, 'DEFAULT_OPTIONS', {}).copy()
    options.update({k: v for k, v in impl_opts.items() if k in options})

    # Create implementation instance inside child
    impl = impl_class(m=m, n=n, k=k, dtype=dtype, **options)

    try:
        # Warmups
        for _ in range(num_warmups):
            impl.run()

        # CUDA timing events
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]

        # Start profiling
        try:
            torch.cuda.cudart().cudaProfilerStart()
        except Exception:
            pass

        last_result = None
        for i in range(num_iterations):
            start_events[i].record()
            last_result = impl.run()
            end_events[i].record()

        # Stop profiling
        try:
            torch.cuda.cudart().cudaProfilerStop()
        except Exception:
            pass

        torch.cuda.synchronize()
        times = [start_events[i].elapsed_time(end_events[i]) for i in range(num_iterations)]
        mean_time = float(_np.mean(times))
        std_time = float(_np.std(times))

        # Include only implementation default option keys in the result row
        default_option_keys = list(getattr(impl_class, 'DEFAULT_OPTIONS', {}).keys())
        impl_option_values = {k: options[k] for k in default_option_keys if k in options}

        result_row = {
            'implementation': impl_id,
            'mean_time (ms)': mean_time,
            'std_time': std_time,
            'min_time': float(min(times)),
            'max_time': float(max(times)),
            'm': m,
            'n': n,
            'k': k,
            'dtype': dtype,
            **impl_option_values,
        }

        if validate and last_result is not None:
            try:
                impl.validate(last_result)
                result_row['valid'] = True
            except Exception as e:
                result_row['valid'] = False
                print(f"Warning: Validation failed for {impl_id} with error: {e}")

        result_queue.put(result_row)
    finally:
        try:
            del impl
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

# Configure pandas to display full output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

class PrimitiveBenchmarkRunner:
    """Main class for running distributed primitive benchmarks."""
    
    ALLOWED_PRIMITIVES = {'tp_columnwise'}
    
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
        if primitive not in self.ALLOWED_PRIMITIVES:
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
    
    # _create_implementation is intentionally omitted in favor of per-impl subprocesses
    
    def run(self) -> pd.DataFrame:
        """
        Run the benchmarks for all implementations.
        
        Returns:
            DataFrame containing benchmark results
        """
        results = []

        # Determine rank without initializing CUDA context
        rank_env = os.environ.get('OMPI_COMM_WORLD_RANK')
        rank = int(rank_env) if rank_env is not None else 0

        # Helper lambda for tqdm iterator creation
        create_tqdm = lambda iterable, **kwargs: tqdm(iterable, **kwargs) if rank == 0 else iterable

        # Prepare multiprocessing context with spawn to isolate CUDA state
        ctx = mp.get_context('spawn')

        for impl_id in create_tqdm(self.implementations, desc="Running benchmarks", position=0):
            if rank == 0:
                print(f"Running benchmark for {impl_id} with options {self.implementation_options[impl_id]}")

            # Spawn isolated child process per implementation
            impl_opts = self.implementation_options.get(impl_id, {})
            result_queue = ctx.SimpleQueue()
            proc = ctx.Process(
                target=_benchmark_worker_entry,
                args=(impl_id, self.m, self.n, self.k, self.dtype, self.num_warmups, self.num_iterations, impl_opts, self.validate, result_queue),
            )
            proc.start()
            result_row = result_queue.get()
            proc.join()

            if rank == 0:
                print(pd.DataFrame([result_row]).to_string(index=False))

            results.append(result_row)

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
            label = f"{impl}"
            # Use known implementation_options to determine which keys to show
            impl_opts = self.implementation_options.get(impl, {})
            display_opts = {k: row[k] for k in impl_opts.keys() if k in row and k != 'implementation'}
            if display_opts:
                label += f" ({', '.join(f'{k}={v}' for k, v in display_opts.items())})"
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