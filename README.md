# Distributed Deep Learning Benchmark (DDLB)

A library for benchmarking distributed deep learning primitives and operations across multiple GPUs.

## Features

- TP-columnwise primitives
- Support for multiple implementations:
  - PyTorch Distributed with various backends (NCCL, UCC/TL)
  - nvFuser with different communications backends and algorithms (including pipelines for comm/compute overlap) 
  - Compute-only reference implementations
- Configurable benchmark parameters via JSON
- Automatic validation of results
- Performance visualization

## Installation

```bash
pip install -e .
```

## Configuration

All benchmark parameters and implementation options are specified in `scripts/config.json`. Example:

```json
{
    "benchmark": {
        "primitive": "tp_columnwise",
        "m": 65536,
        "n": 1024,
        "k": 1024,
        "dtype": "float32",
        "validate": true,
        "num_iterations": 50,
        "num_warmups": 5,
        "implementations": {
            "compute_only": [
                { "size": "unsharded" }
            ],
            "pytorch": [
                {
                    "backend": ["nccl", "ucc/tl/nccl", "ucc/tl/cuda"],
                    "order": ["AG_before", "AG_after"]
                }
            ],
            "fuser": [
                {
                    "algorithm": ["default"],
                    "backend": ["nccl"]
                },
                {
                    "algorithm": ["coll_pipeline"],
                    "s": [2, 8],
                    "backend": ["ucc/tl/nccl"]
                }
            ],
        }
    }
}
```
- You can specify lists for any option to automatically benchmark all combinations (cartesian product).
- Remove or comment out backends that are not supported on your system (e.g., `ucc/tl/ucp` if you do not have UCX/InfiniBand).

## Usage

### Running the Benchmark

```bash
mpirun -np 2 python scripts/run_benchmark.py
```
- Replace `2` with the number of processes/GPUs you want to use.
- Make sure your environment is set up for MPI and CUDA.

### Output
- Results and progress are printed to the console.
- Only rank 0 prints and plots results.

## Troubleshooting

- **MPI errors:** Always use `mpirun` or `mpiexec` to launch the script for distributed runs.
- **UCX/InfiniBand errors:** If you see errors about missing UCX transports or segmentation faults, remove `ucc/tl/ucp` from your config. We have disabled UCX's cuda transport because of a known hang issue.
