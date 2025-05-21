"""
Example script demonstrating how to use the benchmark library with MPI
Supports benchmarking of distributed primitives (currently TP Columnwise)
"""

import os
import torch
import argparse
from ddlb import PrimitiveBenchmarkRunner
from ddlb.communicator import Communicator

def run_benchmark(comm, primitive, m, n, k):
    """Run benchmark for the specified primitive"""
    rank = comm.rank
    world_size = comm.world_size

    if rank == 0:
        print(f"Running {primitive} benchmark with {world_size} MPI processes")

    # Initialize benchmark runner
    runner = PrimitiveBenchmarkRunner(
        primitive=primitive,
        m=m,
        n=n,
        k=k,
        implementations=['pytorch'],
        dtype='float32',
        validate=True,
        num_iterations=5,
        num_warmups=2
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
        # Reorder columns to have TFLOPS first
        cols = ['implementation', 'Throughput (TFLOPS)', 'mean_time (ms)', 'std_time', 'min_time', 'max_time']
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
    args = parser.parse_args()

    # Initialize communicator (handles env vars, CUDA, and process group)
    comm = Communicator()

    # Run the benchmark
    run_benchmark(comm, args.primitive, args.m, args.n, args.k)

if __name__ == '__main__':
    main() 