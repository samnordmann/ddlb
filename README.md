# Distributed Deep Learning Benchmark (DDLB)

A library for benchmarking distributed deep learning primitives and operations across multiple GPUs.

## Features

- Distributed matrix multiplication primitives
- Support for multiple implementations:
  - PyTorch Distributed with various backends (NCCL, UCC/TL)
  - Compute-only reference implementations
- Configurable benchmark parameters via JSON
- Automatic validation of results
- Performance visualization

## Installation

```bash
pip install -e .
```

## Project Structure

```
.
├── README.md
├── config.json
├── ddlb/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── communicator.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── benchmark.py
│   └── primitives/
│       └── TPColumnwise/
│           ├── __init__.py
│           ├── pytorch_tp_columnwise.py
│           ├── compute_only_tp_columnwise.py
│           └── reference_tp_columnwise.py
└── scripts/
    └── run_benchmark.py
```

## Configuration

All benchmark parameters and implementation options are specified in `config.json` at the project root. Example:

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
            ]
        }
    }
}
```
- You can specify lists for any option to automatically benchmark all combinations (cartesian product).
- Remove or comment out backends that are not supported on your system (e.g., `ucc/tl/ucp` if you do not have UCX/InfiniBand).

## Usage

### Running the Benchmark

**With MPI (recommended for distributed runs):**

```bash
mpirun -np 2 python scripts/run_benchmark.py
```
- Replace `2` with the number of processes/GPUs you want to use.
- Make sure your environment is set up for MPI and CUDA.

**Directly (for single-process debugging):**

```bash
python scripts/run_benchmark.py
```

### Output
- Results and progress are printed to the console.
- Only rank 0 prints and plots results.
- If a backend is not supported, you may see errors or crashes—edit `config.json` to remove problematic backends.
