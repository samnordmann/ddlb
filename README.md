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

You can configure benchmarks either:

- via command-line flags (no JSON required), or
- via a JSON file like `scripts/config.json`.

Example JSON:

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

### Command-line (no JSON)

Run directly via the CLI module and pass all parameters as flags.

Quick smoke test (single process):
```bash
mpirun -np 4 python ddlb/cli/benchmark.py \
  --primitive tp_columnwise \
  -m 1024 \
  -n 128 \
  -k 1024 \
  --dtype float16 \
  --num-iterations 5 \
  --num-warmups 2 \
  --impl pytorch;backend=nccl
```

Multiple sizes and implementations:
```bash
mpirun -np 4 python ddlb/cli/benchmark.py \
  --primitive tp_columnwise \
  -m 1024,8192,16384 \
  -n 128,1024,16384 \
  -k 1024,8192,16384 \
  --dtype float16 \
  --num-iterations 50 \
  --num-warmups 5 \
  --impl compute_only;size=sharded,unsharded \
  --impl pytorch;backend=nccl;order=AG_before,AG_after \
  --impl fuser;algorithm=default;backend=nccl;order=AG_before \
  --impl fuser;algorithm=p2p_pipeline;backend=cuda;order=AG_before \
  --impl fuser;algorithm=coll_pipeline;backend=ucc/tl/nccl;order=AG_before;s=8 \
  --impl transformer_engine \
  --impl jax
```

Custom CSV path to write output (supports {timestamp}):
```bash
mpirun -np 4 python ddlb/cli/benchmark.py \
  --primitive tp_columnwise \
  -m 8192 -n 1024 -k 8192 \
  --output-csv results/tp_columnwise_results_{timestamp}.csv \
  --impl pytorch;backend=nccl
```


### JSON-based usage

You can keep complex configurations in a JSON file and run:
```bash
python scripts/run_benchmark.py                # uses scripts/config.json by default
python scripts/run_benchmark.py path/to/config.json
```

With MPI:
```bash
mpirun -np 2 python scripts/run_benchmark.py
mpirun -np 8 python scripts/run_benchmark.py path/to/config.json
```

If you want to generate an Nsight profile, you can use a command like:
```bash
nsys profile --stats=false -w true -t cublas,cuda,nvtx,osrt,mpi,ucx \
  -o /tmp/ddlb_$(date '+%Y-%m-%d_%H-%M-%S') \
  --capture-range-end repeat --capture-range=cudaProfilerApi \
  mpirun -np 8 python scripts/run_benchmark.py
```
Each iteration will produce a separate `nsys` file.

### Programmatic usage (Python)

You can invoke the benchmark runner from Python by constructing the same `config` structure used in JSON:
```python
from ddlb.cli import run_benchmark

config = {
    "benchmark": {
        "primitive": "tp_columnwise",
        "m": [8192],
        "n": [1024],
        "k": [8192],
        "dtype": "float16",
        "validate": True,
        "num_iterations": 10,
        "num_warmups": 3,
        "output_csv": "results/tp_columnwise_results_{timestamp}.csv",
        "implementations": {
            "pytorch": [
                {"backend": "nccl"}
            ]
        }
    }
}

run_benchmark(config)
```

### Output
- Results and progress are printed to the console.
- Only rank 0 prints and plots results.
 - When `output_csv` is provided, a CSV is written; `{timestamp}` is replaced at runtime.

## Troubleshooting

- **MPI errors:** Always use `mpirun` or `mpiexec` to launch the script for distributed runs.
- **UCX/InfiniBand errors:** If you see errors about missing UCX transports or segmentation faults, remove `ucc/tl/ucp` from your config. We have disabled UCX's cuda transport because of a known hang issue.
