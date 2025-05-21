# Distributed Deep Learning Benchmark (DDLB)

A library for benchmarking distributed deep learning primitives and operations across multiple GPUs.

## Features

- Distributed matrix multiplication primitives
- Support for multiple implementations:
  - PyTorch Distributed with various backends (NCCL, UCC/TL)
  - Compute-only reference implementations
- Configurable benchmark parameters
- Automatic validation of results
- Performance visualization

## Installation

```bash
pip install ddlb
```
or 
```
pip install -e .
```
## Usage

### Basic Usage

```python
from ddlb import PrimitiveBenchmarkRunner
from ddlb.communicator import Communicator

# Initialize distributed environment
comm = Communicator()

# Create benchmark instance
runner = PrimitiveBenchmarkRunner(
    primitive='tp_columnwise',
    m=8192,
    n=4096,
    k=8192,
    implementations=['pytorch', 'compute_only'],
    dtype='float32',
    validate=True
)

# Run benchmark and plot results
results = runner.run()
runner.plot_results(results)
```

### Configuration

Benchmarks can be configured using a JSON file. The configuration supports different backends (NCCL, UCC/TL) and communication orders (AG_before, AG_after). See `examples/benchmark_config.json` for available options and `examples/benchmark.py` for usage examples.

### Running with MPI

The benchmark can be run using MPI:

```bash
mpirun -np 8 python examples/benchmark.py
```

## Command Line Interface

```bash
python -m ddlb.examples.benchmark \
    --primitive tp_columnwise \
    --m 8192 \
    --n 4096 \
    --k 8192 \
    --config examples/benchmark_config.json
```

## Development

The project uses a devcontainer for development with CUDA support, PyTorch, and required VS Code extensions.

## License

MIT 