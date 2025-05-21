"""
Example script demonstrating how to use the benchmark library with MPI
Supports benchmarking of distributed primitives (currently TP Columnwise)
"""

import argparse
import json
import os
import torch
from ddlb import PrimitiveBenchmarkRunner
from ddlb.communicator import Communicator

def run_benchmark(comm: Communicator, primitive: str, m: int, n: int, k: int, config: dict) -> None:
    """Run benchmark for the specified primitive with all configurations.
    
    Args:
        comm: Communicator instance for distributed environment
        primitive: Name of the primitive to benchmark
        m: Number of rows in first matrix
        n: Number of columns in second matrix
        k: Number of columns in first matrix / rows in second matrix
        config: Dictionary of implementation configurations
    """
    rank = comm.rank
    world_size = comm.world_size

    if rank == 0:
        print(f"Running {primitive} benchmark with {world_size} MPI processes")
        print("\nConfigurations:")
        for impl_name, impl_configs in config.items():
            for i, opts in enumerate(impl_configs):
                print(f"  {impl_name}_{i}: {opts}")

    # Create list of implementations with their configurations
    implementations = []
    implementation_options = {}
    
    for impl_name, impl_configs in config.items():
        for i, opts in enumerate(impl_configs):
            impl_id = f"{impl_name}_{i}"
            implementations.append(impl_id)
            implementation_options[impl_id] = {
                'implementation': impl_name,
                **opts
            }

    # Initialize and run benchmark
    runner = PrimitiveBenchmarkRunner(
        primitive=primitive,
        m=m,
        n=n,
        k=k,
        implementations=implementations,
        dtype='float32',
        validate=True,
        num_iterations=100,
        num_warmups=10,
        implementation_options=implementation_options
    )
    results = runner.run()

    # Only rank 0 prints and plots results
    if rank == 0:
        print("\nBenchmark Results:")
        print(f"Matrix dimensions: ({m}, {n}, {k})")
        print(f"Total matrix size: {m*n + n*k + m*k} elements")
        print("\nDetailed Results:")
        
        # Calculate TFLOPS (2 FLOPs per multiply-add)
        total_flops = 2 * m * n * k
        results['Throughput (TFLOPS)'] = total_flops / (results['mean_time (ms)'] * 1e9)
        # compute the interval error as two times the standard deviation of the throughput
        results['Throughput Interval error'] = 2 * total_flops * results['std_time'] / (results['mean_time (ms)']**2 * 1e9) 
        
        # Create a more readable implementation name that includes options
        def format_config(x):
            opts = implementation_options[x]
            impl = opts['implementation']
            other_opts = {k: v for k, v in opts.items() if k != 'implementation'}
            if other_opts:
                return f"{impl} ({', '.join(f'{k}={v}' for k, v in other_opts.items())})"
            return impl
        
        results['config'] = results['implementation'].apply(format_config)
        
        # Format throughput with standard deviation
        def format_throughput(row):
            mean = round(row['Throughput (TFLOPS)'], 1)
            error = round(row['Throughput Interval error'], 1)
            return f"{mean} (±{error})"
        
        results['Throughput (TFLOPS)'] = results.apply(format_throughput, axis=1)
        
        # Display and plot results
        cols = ['config', 'Throughput (TFLOPS)', 'mean_time (ms)', 'std_time', 'min_time', 'max_time']
        print(results[cols])
        runner.plot_results(results)

def main() -> None:
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description='Run distributed primitive benchmarks')
    parser.add_argument('--primitive', type=str, default='tp_columnwise',
                      help='Primitive to benchmark (currently only tp_columnwise supported)')
    parser.add_argument('--m', type=int, default=8192, 
                      help='Number of rows in first matrix')
    parser.add_argument('--n', type=int, default=4096, 
                      help='Number of columns in second matrix')
    parser.add_argument('--k', type=int, default=8192, 
                      help='Number of columns in first matrix / rows in second matrix')
    parser.add_argument('--config', type=str, default='examples/benchmark_config.json',
                      help='Path to JSON configuration file')
    args = parser.parse_args()

    # Load configuration from JSON file
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return

    # Initialize communicator and run benchmark
    comm = Communicator()
    run_benchmark(comm, args.primitive, args.m, args.n, args.k, config)
    del comm

if __name__ == '__main__':
    main() 