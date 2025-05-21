# Distributed Deep Learning Benchmark (DDLB)

A library for benchmarking different distributed deep learning implementations.

## Features

- PyTorch Distributed
- Transformer Engine
- NCCL
- Fuser

## Installation

```bash
pip install ddlb
```

## Usage

```python
from ddlb import BenchmarkRunner
from ddlb.communicator import Communicator

# Initialize distributed environment
comm = Communicator()

# Create benchmark instance
runner = BenchmarkRunner(
    m=8192,  # rows of first matrix
    n=4096,  # columns of second matrix
    k=8192,  # columns of first matrix / rows of second matrix
    num_gpus=comm.world_size,
    implementations=['pytorch_distributed', 'transformer_engine']
)

# Run benchmark
results = runner.run()
```

## License

MIT 