"""Benchmark CLI functionality."""

import json
import os
import itertools
from typing import Dict, List, Any, Optional
from ddlb import PrimitiveBenchmarkRunner

def generate_config_combinations(config: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Generate all possible combinations of configuration parameters.
    
    Args:
        config: Dictionary of implementation configurations where values can be lists
        
    Returns:
        Dictionary with all possible combinations of configurations
    """
    expanded_config = {}
    
    for impl_name, impl_configs in config.items():
        expanded_config[impl_name] = []
        
        for base_config in impl_configs:
            # Find all list parameters
            list_params = {k: v for k, v in base_config.items() if isinstance(v, list)}
            if not list_params:
                # If no list parameters, just add the config as is
                expanded_config[impl_name].append(base_config)
                continue
                
            # Generate cartesian product of all list parameters
            param_names = list(list_params.keys())
            param_values = list(list_params.values())
            
            for combination in itertools.product(*param_values):
                # Create new config with this combination
                new_config = base_config.copy()
                for name, value in zip(param_names, combination):
                    new_config[name] = value
                expanded_config[impl_name].append(new_config)
    
    return expanded_config

def run_benchmark(config: dict) -> None:
    """Run benchmark for the specified primitive with all configurations.
    
    Args:
        config: Dictionary containing benchmark and implementation configurations
    """
    # Read rank/world_size from MPI/SLURM env directly to avoid initializing CUDA in parent
    rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 
                         os.getenv("SLURM_PROCID", 
                                  os.getenv("PMI_RANK", "0"))))
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", 
                               os.getenv("SLURM_NTASKS", 
                                        os.getenv("PMI_SIZE", "1"))))

    # Extract benchmark parameters
    benchmark_config = config['benchmark']
    primitive = benchmark_config['primitive']
    m = benchmark_config['m']
    n = benchmark_config['n']
    k = benchmark_config['k']
    dtype = benchmark_config['dtype']
    validate = benchmark_config['validate']
    num_iterations = benchmark_config['num_iterations']
    num_warmups = benchmark_config['num_warmups']

    # Generate all possible combinations of configurations
    expanded_config = generate_config_combinations(benchmark_config['implementations'])

    if rank == 0:
        print(f"Running {primitive} benchmark with {world_size} MPI processes")
        print(f"Matrix dimensions: ({m}, {n}, {k})")
        print(f"Total matrix size: {m*n + n*k + m*k} elements")
        print("\nConfigurations:")
        for impl_name, impl_configs in expanded_config.items():
            for i, opts in enumerate(impl_configs):
                print(f"  {impl_name}_{i}: {opts}")

    # Create list of implementations with their configurations
    implementations = []
    implementation_options = {}
    
    for impl_name, impl_configs in expanded_config.items():
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
        dtype=dtype,
        validate=validate,
        num_iterations=num_iterations,
        num_warmups=num_warmups,
        implementation_options=implementation_options
    )
    results = runner.run()

    # Only rank 0 prints and plots results
    if rank == 0:
        print("\nBenchmark Results:")
        
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