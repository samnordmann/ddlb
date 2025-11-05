"""Benchmark CLI functionality."""

import json
import os
import itertools
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
from ddlb import PrimitiveBenchmarkRunner
import argparse


def _infer_scalar(value: str) -> Any:
    """Infer basic Python type from a string token.

    Supports int, float, bool, and str as fallback.
    """
    v = value.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        if v.startswith("0") and v != "0":
            # preserve strings like 08 rather than interpreting as octal/ints
            raise ValueError
        return int(v)
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return v


def _parse_value_list(v: str) -> Any:
    """Parse a comma-separated value list with type inference.

    Returns a list when multiple tokens are present; otherwise a scalar.
    """
    parts = [p for p in (v or "").split(",")]
    parts = [p.strip() for p in parts if len(p.strip()) > 0]
    if len(parts) == 0:
        return ""
    if len(parts) == 1:
        return _infer_scalar(parts[0])
    return [_infer_scalar(p) for p in parts]


def _parse_int_list(v: str) -> List[int]:
    """Parse a comma-separated integer list from a string."""
    values = [s.strip() for s in str(v).split(",") if len(s.strip()) > 0]
    return [int(x) for x in values]


def _parse_impl_spec(spec: str) -> (str, Dict[str, Any]):
    """Parse implementation spec of the form:

    name;key=value[,value];key2=value

    - Multiple --impl flags can be used; each becomes a separate base config.
    - Values separated by commas become a list; scalars are inferred.
    """
    if spec is None:
        raise ValueError("Empty implementation spec")
    tokens = [t for t in str(spec).split(";") if len(t.strip()) > 0]
    if len(tokens) == 0:
        raise ValueError("Invalid implementation spec: empty")
    name = tokens[0].strip()
    options: Dict[str, Any] = {}
    for tok in tokens[1:]:
        if "=" not in tok:
            # treat as flag-like boolean true
            k = tok.strip()
            if k:
                options[k] = True
            continue
        k, v = tok.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        options[k] = _parse_value_list(v)
    return name, options

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

        # The 'implementation' column is already a human-readable label including options
        results['config'] = results['implementation']
        
        # Display results (aggregated across shapes)
        cols = ['m', 'n', 'k', 'config', 'Throughput (TFLOPS)', 'Throughput std (TFLOPS)', 'mean_time (ms)', 'std_time', 'min_time', 'max_time']
        print(results[cols])


def main() -> None:
    """CLI entry point to run benchmarks without a JSON file."""
    parser = argparse.ArgumentParser(description="Run DDLB benchmarks via command line arguments")
    parser.add_argument(
        "--primitive",
        required=True,
        choices=["tp_columnwise"],
        help="Primitive to benchmark",
    )
    parser.add_argument(
        "-m",
        "--m",
        required=True,
        help="Comma-separated list of m sizes (e.g., 1024,8192,16384)",
    )
    parser.add_argument(
        "-n",
        "--n",
        required=True,
        help="Comma-separated list of n sizes (e.g., 128,1024,16384)",
    )
    parser.add_argument(
        "-k",
        "--k",
        required=True,
        help="Comma-separated list of k sizes (e.g., 1024,8192,16384)",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        help="Data type (e.g., float16, float32)",
    )
    parser.add_argument(
        "--validate",
        dest="validate",
        action="store_true",
        default=True,
        help="Enable result validation (default)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=50,
        help="Number of timing iterations",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional output CSV path; supports {timestamp}",
    )
    parser.add_argument(
        "--impl",
        action="append",
        required=True,
        help=(
            "Implementation spec: name;key=value[,value];key2=value. "
            "Repeat --impl to add multiple base configs."
        ),
    )

    args = parser.parse_args()

    # Build implementations mapping: name -> list[options dict]
    implementations: Dict[str, List[Dict[str, Any]]] = {}
    for spec in args.impl:
        name, opts = _parse_impl_spec(spec)
        implementations.setdefault(name, []).append(opts)

    config: Dict[str, Any] = {
        "benchmark": {
            "primitive": args.primitive,
            "m": _parse_int_list(args.m),
            "n": _parse_int_list(args.n),
            "k": _parse_int_list(args.k),
            "dtype": args.dtype,
            "validate": args.validate,
            "num_iterations": int(args.num_iterations),
            "num_warmups": int(args.num_warmups),
            "output_csv": args.output_csv,
            "implementations": implementations,
        }
    }

    run_benchmark(config)


if __name__ == "__main__":
    main()