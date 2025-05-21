"""
Example script demonstrating how to use the benchmark library with MPI
Supports benchmarking of distributed primitives (currently TP Columnwise)
"""

import os
import torch
import argparse
import json
from ddlb import PrimitiveBenchmarkRunner
from ddlb.communicator import Communicator

def run_benchmark(comm, primitive, m, n, k, config):
    """Run benchmark for the specified primitive with all configurations"""
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

    # Initialize benchmark runner
    runner = PrimitiveBenchmarkRunner(
        primitive=primitive,
        m=m,
        n=n,
        k=k,
        implementations=implementations,
        dtype='float32',
        validate=True,
        num_iterations=5,
        num_warmups=2,
        implementation_options=implementation_options
    )

    # Run benchmarks
    results = runner.run()

    # Only rank 0 prints and plots results
    if rank == 0:
        print("\nBenchmark Results:")
        print(f"Matrix dimensions: ({m}, {n}, {k})")
        print(f"Total matrix size: {m*n + n*k + m*k} elements")
        print("\nDetailed Results:")
        # Calculate TFLOPS
        total_flops = 2 * m * n * k  # 2 FLOPs per multiply-add
        results['Throughput (TFLOPS)'] = (total_flops / (results['mean_time (ms)'] * 1e9))  # ms to s, so 1e9
        
        # Create a more readable implementation name that includes options
        results['config'] = results['implementation'].apply(
            lambda x: f"{implementation_options[x]['implementation']} ({', '.join(f'{k}={v}' for k, v in implementation_options[x].items() if k != 'implementation')})"
        )
        
        # Reorder columns to have TFLOPS first
        cols = ['config', 'Throughput (TFLOPS)', 'mean_time (ms)', 'std_time', 'min_time', 'max_time']
        print(results[cols])
        runner.plot_results(results)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run distributed primitive benchmarks')
    parser.add_argument('--primitive', type=str, default='tp_columnwise',
                      help='Primitive to benchmark (currently only tp_columnwise supported)')
    parser.add_argument('--m', type=int, default=8192, help='Number of rows in first matrix')
    parser.add_argument('--n', type=int, default=4096, help='Number of columns in second matrix')
    parser.add_argument('--k', type=int, default=8192, help='Number of columns in first matrix / rows in second matrix')
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

    # Initialize communicator (handles env vars, CUDA, and process group)
    comm = Communicator()

    # Run the benchmark with all configurations
    run_benchmark(comm, args.primitive, args.m, args.n, args.k, config)

if __name__ == '__main__':
    main() 