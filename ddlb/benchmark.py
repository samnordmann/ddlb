"""
Main benchmark runner module for distributed primitives
"""

import os
import csv
import time
import multiprocessing as mp
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ddlb.envs import get_world_size, get_rank

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
    time_measurement_backend: str = 'cpu_clock',
    barrier_at_each_iteration: bool = True,
    primitive: str = 'tp_rowwise'
):
    # Import CUDA-dependent libraries inside child process only and lazily
    import torch
    import torch.distributed as dist
    import numpy as _np
    import importlib as _importlib
    import socket as _socket

    def _load_impl_class(base_impl: str, primitive: str):
        # Map primitive and implementation to submodule path and class name
        primitive_mappings = {
            'tp_columnwise': {
                'pytorch': ('ddlb.primitives.TPColumnwise.pytorch', 'PyTorchTPColumnwise'),
                'pytorch_sym_mem': ('ddlb.primitives.TPColumnwise.pytorch_sym_mem', 'PyTorchSymMemTPColumnwise'),
                'compute_only': ('ddlb.primitives.TPColumnwise.compute_only', 'ComputeOnlyTPColumnwise'),
                'fuser': ('ddlb.primitives.TPColumnwise.fuser', 'FuserTPColumnwise'),
                'transformer_engine': ('ddlb.primitives.TPColumnwise.transformer_engine', 'TransformerEngineTPColumnwise'),
                'jax': ('ddlb.primitives.TPColumnwise.jax_tp', 'JAXTPColumnwise'),
                'ddlp': ('ddlb.primitives.TPColumnwise.ddlp', 'DDLPTPColumnwise'),
            },
            'tp_rowwise': {
                'pytorch': ('ddlb.primitives.TPRowwise.pytorch', 'PyTorchTPRowwise'),
                'fuser': ('ddlb.primitives.TPRowwise.fuser', 'FuserTPRowwise'),
                'transformer_engine': ('ddlb.primitives.TPRowwise.transformer_engine', 'TransformerEngineTPRowwise'),
            },
        }
        
        if primitive not in primitive_mappings:
            raise ValueError(f"Unknown primitive: {primitive}")
        
        mapping = primitive_mappings[primitive]
        if base_impl not in mapping:
            raise ValueError(f"Unknown implementation '{base_impl}' for primitive '{primitive}'")
        
        module_path, class_name = mapping[base_impl]
        module = _importlib.import_module(module_path)
        return getattr(module, class_name)

    # Parse base implementation and options
    if '_' in impl_id:
        base_impl = '_'.join(impl_id.split('_')[:-1])
    else:
        base_impl = impl_id

    impl_class = _load_impl_class(base_impl, primitive)
    options = getattr(impl_class, 'DEFAULT_OPTIONS', {}).copy()
    options.update({k: v for k, v in impl_opts.items() if k in options})

    # Create implementation instance inside child
    impl = impl_class(m=m, n=n, k=k, dtype=dtype, **options)

    try:
        # Warmups
        for _ in range(num_warmups):
            impl.run()

        torch.cuda.synchronize()

        # Start profiling
        try:
            torch.cuda.cudart().cudaProfilerStart()
        except Exception:
            pass

        for i in range(5):
            impl.run()

        torch.cuda.synchronize()

        # Stop profiling
        try:
            torch.cuda.cudart().cudaProfilerStop()
        except Exception:
            pass

        # Include only implementation default option keys in the result row
        default_option_keys = list(getattr(impl_class, 'DEFAULT_OPTIONS', {}).keys())
        impl_option_values = {k: options[k] for k in default_option_keys if k in options}
        # Exclude size from CSV columns and labels
        filtered_impl_option_values = {k: v for k, v in impl_option_values.items() if k != 'size'}

        # Human-readable implementation label with options (excluding size)
        impl_label = base_impl
        if filtered_impl_option_values:
            impl_label += f" (" + ", ".join(f"{k}={v}" for k, v in filtered_impl_option_values.items()) + ")"

        # Consolidate implementation options into a single string column 'option'
        ordered_option_keys = [k for k in getattr(impl_class, 'DEFAULT_OPTIONS', {}).keys() if k in filtered_impl_option_values]
        option_str = ", ".join(f"{k}={filtered_impl_option_values[k]}" for k in ordered_option_keys)

        for i in range(num_warmups):
            impl.run()

        # Timing loop based on backend configuration
        times = []
        
        if time_measurement_backend == 'cuda_event':
            if barrier_at_each_iteration:
                start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
                end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
                
                for i in range(num_iterations):
                    # NCCL allreduce + wait
                    dummy = torch.tensor([0], device=impl.communicator.device, dtype=torch.int)
                    if dist.is_initialized():
                        dist.all_reduce(dummy)
                        torch.cuda.synchronize()
                    
                    start_events[i].record()
                    last_result = impl.run()
                    end_events[i].record()
                
                torch.cuda.synchronize()
                times = [start_events[i].elapsed_time(end_events[i]) for i in range(num_iterations)]
            else:
                 # Aggregate measurement without barrier
                 start_event = torch.cuda.Event(enable_timing=True)
                 end_event = torch.cuda.Event(enable_timing=True)
                 
                 torch.cuda.synchronize()
                 start_event.record()
                 for i in range(num_iterations):
                     last_result = impl.run()
                 end_event.record()
                 
                 torch.cuda.synchronize()
                 total_time_ms = start_event.elapsed_time(end_event)
                 mean_time = float(total_time_ms / num_iterations)
                 times = [mean_time] * num_iterations
            
        elif time_measurement_backend == 'cpu_clock':
            if barrier_at_each_iteration:
                # Per-iteration measurement with barrier
                for i in range(num_iterations):
                    impl.communicator.barrier()
                    
                    start_time = time.perf_counter()
                    last_result = impl.run()
                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    
                    times.append((end_time - start_time) * 1000)
            else:
                # Aggregate measurement without barrier
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                for i in range(num_iterations):
                    last_result = impl.run()
                    
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                total_time_ms = (end_time - start_time) * 1000
                mean_time = float(total_time_ms / num_iterations)
                times = [mean_time] * num_iterations
        else:
            raise ValueError(f"Unknown time_measurement_backend: {time_measurement_backend}")

        # AllReduce times vector with MAX operator across ranks
        times_tensor = torch.tensor(times, device=impl.communicator.device, dtype=torch.float64)
        if not dist.is_initialized():
             # Initialize process group if not already initialized
             master_addr = os.environ.get('DDLB_MASTER_ADDR', 'localhost')
             master_port = os.environ.get('DDLB_MASTER_PORT', '12345')
             dist.init_process_group(
                backend='nccl',
                rank=impl.communicator.rank,
                world_size=impl.communicator.world_size,
                init_method=f"tcp://{master_addr}:{master_port}",
                device_id=impl.communicator.device
            )
        dist.all_reduce(times_tensor, op=dist.ReduceOp.MAX)
        times = times_tensor.tolist()
        
        mean_time = float(_np.mean(times))
        std_time = float(_np.std(times))

        # Compute throughput metrics (TFLOPS) per-iteration and aggregate
        # Constant converts ms to seconds and FLOPs to TFLOPs: 2*m*n*k / (ms * 1e9)
        thr_const = (2.0 * m * n * k) / 1e9
        throughputs = [thr_const / t for t in times if t > 0]
        mean_throughput = float(_np.mean(throughputs)) if throughputs else 0.0
        std_throughput = float(_np.std(throughputs)) if throughputs else 0.0

        # MPI world size and hostname for traceability
        world_size = get_world_size()
        hostname = _socket.gethostname()

        result_row = {
            'implementation': impl_label,
            'mean_time (ms)': mean_time,
            'std_time': std_time,
            'min_time': float(min(times)),
            'max_time': float(max(times)),
            'm': m,
            'n': n,
            'k': k,
            'dtype': dtype,
            'Throughput (TFLOPS)': mean_throughput,
            'Throughput std (TFLOPS)': std_throughput,
            'world_size': world_size,
            'hostname': hostname,
            'time_measurement_backend': time_measurement_backend,
            'barrier_at_each_iteration': barrier_at_each_iteration,
            'option': option_str,
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
    
    ALLOWED_PRIMITIVES = {'tp_columnwise', 'tp_rowwise'}
    
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
        output_csv: Optional[str] = None,
        time_measurement_backend: str = 'cpu_clock',
        barrier_at_each_iteration: bool = True,
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            primitive: Name of the primitive to benchmark ('tp_columnwise' or 'tp_rowwise')
            m: Number of rows in first matrix
            n: Number of columns in second matrix
            k: Number of columns in first matrix / rows in second matrix
            implementations: List of implementation names to benchmark
            dtype: Data type for the matrices
            validate: Whether to validate results
            num_iterations: Number of iterations for timing
            num_warmups: Number of warmup iterations
            implementation_options: Optional dictionary of implementation-specific options
            time_measurement_backend: Backend for timing ('cpu_clock' or 'cuda_event')
            barrier_at_each_iteration: Whether to synchronize ranks before each iteration
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
        self.output_csv = output_csv
        self.time_measurement_backend = time_measurement_backend
        self.barrier_at_each_iteration = barrier_at_each_iteration
    
    # _create_implementation is intentionally omitted in favor of per-impl subprocesses
    
    def run(self) -> pd.DataFrame:
        """
        Run the benchmarks for all implementations.
        
        Returns:
            DataFrame containing benchmark results
        """
        results = []

        # Determine rank without initializing CUDA context
        rank = get_rank()

        # Helper lambda for tqdm iterator creation
        create_tqdm = lambda iterable, **kwargs: tqdm(iterable, **kwargs) if rank == 0 else iterable

        # Prepare multiprocessing context with spawn to isolate CUDA state
        ctx = mp.get_context('spawn')

        # Setup CSV output (rank 0 only). Create default path if not provided.
        output_csv_path: Optional[str] = None
        if rank == 0:
            if self.output_csv and len(str(self.output_csv).strip()) > 0:
                output_csv_path = self.output_csv

            # Ensure directory exists
            try:
                os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            except Exception:
                pass

            # Write header if file doesn't exist yet
            if output_csv_path and not os.path.exists(output_csv_path):
                # Minimal header; full headers written on first row write
                with open(output_csv_path, 'a') as _:
                    pass

        for impl_id in create_tqdm(self.implementations, desc="Running benchmarks", position=0):
            if rank == 0:
                print(f"Running benchmark for {impl_id} with options {self.implementation_options[impl_id]}")

            # Spawn isolated child process per implementation
            impl_opts = self.implementation_options.get(impl_id, {})
            result_queue = ctx.SimpleQueue()
            proc = ctx.Process(
                target=_benchmark_worker_entry,
                args=(impl_id, self.m, self.n, self.k, self.dtype, self.num_warmups, self.num_iterations, impl_opts, self.validate, result_queue, self.time_measurement_backend, self.barrier_at_each_iteration, self.primitive),
            )
            proc.start()
            result_row = result_queue.get()
            proc.join()

            if rank == 0:
                print(pd.DataFrame([result_row]).to_string(index=False))

                # Append row to CSV immediately to avoid losing progress
                if output_csv_path:
                    df_row = pd.DataFrame([result_row])
                    # On first write, include header if file is empty
                    write_header = False
                    try:
                        write_header = os.path.getsize(output_csv_path) == 0
                    except Exception:
                        write_header = True
                    df_row.to_csv(output_csv_path, mode='a', index=False, header=write_header, quoting=csv.QUOTE_MINIMAL)

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