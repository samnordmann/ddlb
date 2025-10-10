"""Benchmark CLI functionality."""

import json
import os
import itertools
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
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
    # Read rank/world_size from MPI env directly to avoid initializing CUDA in parent
    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
    world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))

    # Extract benchmark parameters
    benchmark_config = config['benchmark']
    primitive = benchmark_config['primitive']
    m_cfg = benchmark_config['m']
    n_cfg = benchmark_config['n']
    k_cfg = benchmark_config['k']
    dtype = benchmark_config['dtype']
    validate = benchmark_config['validate']
    num_iterations = benchmark_config['num_iterations']
    num_warmups = benchmark_config['num_warmups']
    output_csv: Optional[str] = benchmark_config.get('output_csv')

    # Generate all possible combinations of configurations
    expanded_config = generate_config_combinations(benchmark_config['implementations'])

    # Normalize m, n, k to lists to support cartesian product across sizes
    to_list = lambda x: x if isinstance(x, list) else [x]
    m_list = to_list(m_cfg)
    n_list = to_list(n_cfg)
    k_list = to_list(k_cfg)

    shapes = list(itertools.product(m_list, n_list, k_list))

    if rank == 0:
        print(f"Running {primitive} benchmark with {world_size} MPI processes")
        print(f"Number of shapes: {len(shapes)}")
        print("Shapes:")
        for (mm, nn, kk) in shapes:
            print(f"  ({mm}, {nn}, {kk})")
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

    # Compute a single output CSV path for this run; support {timestamp} token
    output_csv_path: Optional[str] = output_csv
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if rank == 0:
        if output_csv_path is not None and len(str(output_csv_path).strip()) > 0:
            output_csv_path = output_csv_path.replace('{timestamp}', run_timestamp)
        else:
            # Use first shape or generic label if shapes empty
            shape_label = f"{m_list[0]}x{k_list[0]}x{n_list[0]}" if len(shapes) > 0 else "shapes"
            output_csv_path = f"results/{primitive}_{shape_label}_{dtype}_{run_timestamp}.csv"

    # Run benchmarks across all requested shapes and aggregate results
    result_frames: List[pd.DataFrame] = []
    for (mm, nn, kk) in shapes:
        if rank == 0:
            print(f"\n--- Running shape ({mm}, {nn}, {kk}) ---")
        runner = PrimitiveBenchmarkRunner(
            primitive=primitive,
            m=mm,
            n=nn,
            k=kk,
            implementations=implementations,
            dtype=dtype,
            validate=validate,
            num_iterations=num_iterations,
            num_warmups=num_warmups,
            implementation_options=implementation_options,
            output_csv=output_csv_path
        )
        result_frames.append(runner.run())

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()

    # Only rank 0 prints and plots results
    if rank == 0:
        print("\nBenchmark Results:")

        # Create a more readable implementation name that includes options
        def format_config(x):
            opts = implementation_options[x]
            impl = opts['implementation']
            other_opts = {k: v for k, v in opts.items() if k != 'implementation'}
            if other_opts:
                return f"{impl} ({', '.join(f'{k}={v}' for k, v in other_opts.items())})"
            return impl
        
        results['config'] = results['implementation'].apply(format_config)
        
        # Display results (aggregated across shapes)
        cols = ['m', 'n', 'k', 'config', 'Throughput (TFLOPS)', 'Throughput std (TFLOPS)', 'mean_time (ms)', 'std_time', 'min_time', 'max_time']
        print(results[cols])